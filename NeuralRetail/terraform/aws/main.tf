# =============================================================================
# NeuralRetail -- Terraform AWS Infrastructure
# Phase 4: Full production IaC
#
# Resources:
#   - VPC + 2 public + 2 private subnets across 2 AZs
#   - EKS cluster + managed node group
#   - RDS PostgreSQL 15 (Multi-AZ)
#   - ElastiCache Redis 7 cluster
#   - S3 bucket for model artifacts + MLflow
#   - AWS Secrets Manager for all credentials
#   - IAM roles for EKS, RDS access, S3 access
# =============================================================================

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # Remote state in S3 -- uncomment for team use
  # backend "s3" {
  #   bucket = "neuralretail-tfstate"
  #   key    = "prod/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project     = "NeuralRetail"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# =============================================================================
# Variables
# =============================================================================

variable "aws_region"      { default = "us-east-1" }
variable "environment"     { default = "production" }
variable "project_name"    { default = "neuralretail" }
variable "db_password"     { sensitive = true }
variable "eks_node_type"   { default = "m5.large" }
variable "eks_min_nodes"   { default = 2 }
variable "eks_max_nodes"   { default = 6 }
variable "eks_desired"     { default = 3 }

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  azs         = ["${var.aws_region}a", "${var.aws_region}b"]
}

# =============================================================================
# VPC
# =============================================================================

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags = { Name = "${local.name_prefix}-vpc" }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${local.name_prefix}-igw" }
}

# Public subnets (EKS nodes, NAT gateways)
resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet("10.0.0.0/16", 8, count.index)
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true
  tags = {
    Name                                        = "${local.name_prefix}-public-${count.index + 1}"
    "kubernetes.io/role/elb"                    = "1"
    "kubernetes.io/cluster/${local.name_prefix}-eks" = "owned"
  }
}

# Private subnets (RDS, ElastiCache)
resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet("10.0.0.0/16", 8, count.index + 10)
  availability_zone = local.azs[count.index]
  tags = {
    Name                                        = "${local.name_prefix}-private-${count.index + 1}"
    "kubernetes.io/role/internal-elb"           = "1"
    "kubernetes.io/cluster/${local.name_prefix}-eks" = "owned"
  }
}

# NAT Gateway for private subnets
resource "aws_eip" "nat" {
  count  = 1
  domain = "vpc"
}

resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public[0].id
  tags          = { Name = "${local.name_prefix}-nat" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
  tags = { Name = "${local.name_prefix}-public-rt" }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat.id
  }
  tags = { Name = "${local.name_prefix}-private-rt" }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# =============================================================================
# Security Groups
# =============================================================================

resource "aws_security_group" "eks_nodes" {
  name        = "${local.name_prefix}-eks-nodes"
  description = "EKS worker nodes"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    self      = true
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "rds" {
  name        = "${local.name_prefix}-rds"
  description = "RDS PostgreSQL access from EKS"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
  }
}

resource "aws_security_group" "redis" {
  name        = "${local.name_prefix}-redis"
  description = "ElastiCache Redis access from EKS"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
  }
}

# =============================================================================
# IAM -- EKS
# =============================================================================

data "aws_iam_policy_document" "eks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "eks_cluster" {
  name               = "${local.name_prefix}-eks-cluster-role"
  assume_role_policy = data.aws_iam_policy_document.eks_assume.json
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  role       = aws_iam_role.eks_cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

data "aws_iam_policy_document" "node_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "eks_nodes" {
  name               = "${local.name_prefix}-eks-node-role"
  assume_role_policy = data.aws_iam_policy_document.node_assume.json
}

resource "aws_iam_role_policy_attachment" "node_worker" {
  for_each = toset([
    "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
    "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy",
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
    "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess",
  ])
  role       = aws_iam_role.eks_nodes.name
  policy_arn = each.value
}

# S3 full access for MLflow / model artifacts
resource "aws_iam_policy" "s3_model_store" {
  name = "${local.name_prefix}-s3-model-store"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
      Resource = [
        aws_s3_bucket.model_store.arn,
        "${aws_s3_bucket.model_store.arn}/*",
      ]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "s3_model_store" {
  role       = aws_iam_role.eks_nodes.name
  policy_arn = aws_iam_policy.s3_model_store.arn
}

# Secrets Manager read access for EKS nodes
resource "aws_iam_policy" "secrets_read" {
  name = "${local.name_prefix}-secrets-read"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"]
      Resource = "arn:aws:secretsmanager:${var.aws_region}:*:secret:${local.name_prefix}/*"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "secrets_read" {
  role       = aws_iam_role.eks_nodes.name
  policy_arn = aws_iam_policy.secrets_read.arn
}

# =============================================================================
# EKS Cluster
# =============================================================================

resource "aws_eks_cluster" "main" {
  name     = "${local.name_prefix}-eks"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.30"

  vpc_config {
    subnet_ids              = concat(aws_subnet.public[*].id, aws_subnet.private[*].id)
    endpoint_private_access = true
    endpoint_public_access  = true
    security_group_ids      = [aws_security_group.eks_nodes.id]
  }

  depends_on = [aws_iam_role_policy_attachment.eks_cluster_policy]
}

resource "aws_eks_node_group" "main" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${local.name_prefix}-ng"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = aws_subnet.private[*].id
  instance_types  = [var.eks_node_type]

  scaling_config {
    min_size     = var.eks_min_nodes
    max_size     = var.eks_max_nodes
    desired_size = var.eks_desired
  }

  update_config { max_unavailable = 1 }

  depends_on = [aws_iam_role_policy_attachment.node_worker]
}

# =============================================================================
# RDS PostgreSQL 15 (Multi-AZ)
# =============================================================================

resource "aws_db_subnet_group" "main" {
  name       = "${local.name_prefix}-db-subnet"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_db_instance" "postgres" {
  identifier              = "${local.name_prefix}-postgres"
  engine                  = "postgres"
  engine_version          = "15.6"
  instance_class          = "db.t3.medium"
  allocated_storage       = 100
  max_allocated_storage   = 500
  storage_encrypted       = true
  db_name                 = "neuralretail"
  username                = "neuralretail"
  password                = var.db_password
  db_subnet_group_name    = aws_db_subnet_group.main.name
  vpc_security_group_ids  = [aws_security_group.rds.id]
  multi_az                = true
  backup_retention_period = 7
  skip_final_snapshot     = false
  final_snapshot_identifier = "${local.name_prefix}-final-snapshot"
  deletion_protection     = true
  performance_insights_enabled = true
  tags = { Name = "${local.name_prefix}-postgres" }
}

# =============================================================================
# ElastiCache Redis 7 (cluster mode)
# =============================================================================

resource "aws_elasticache_subnet_group" "main" {
  name       = "${local.name_prefix}-redis-subnet"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "${local.name_prefix}-redis"
  description                = "NeuralRetail Feast online store + session cache"
  node_type                  = "cache.t3.medium"
  port                       = 6379
  parameter_group_name       = "default.redis7"
  num_cache_clusters         = 2
  automatic_failover_enabled = true
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  subnet_group_name          = aws_elasticache_subnet_group.main.name
  security_group_ids         = [aws_security_group.redis.id]
  tags = { Name = "${local.name_prefix}-redis" }
}

# =============================================================================
# S3 -- Model artifacts + MLflow tracking
# =============================================================================

resource "aws_s3_bucket" "model_store" {
  bucket        = "${local.name_prefix}-model-store"
  force_destroy = false
  tags          = { Name = "${local.name_prefix}-model-store" }
}

resource "aws_s3_bucket_versioning" "model_store" {
  bucket = aws_s3_bucket.model_store.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_store" {
  bucket = aws_s3_bucket.model_store.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "model_store" {
  bucket = aws_s3_bucket.model_store.id
  rule {
    id     = "expire-old-model-versions"
    status = "Enabled"
    noncurrent_version_expiration { noncurrent_days = 90 }
  }
}

resource "aws_s3_bucket_public_access_block" "model_store" {
  bucket                  = aws_s3_bucket.model_store.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# =============================================================================
# AWS Secrets Manager
# =============================================================================

resource "aws_secretsmanager_secret" "db_credentials" {
  name                    = "${local.name_prefix}/db-credentials"
  description             = "NeuralRetail RDS PostgreSQL credentials"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = "neuralretail"
    password = var.db_password
    host     = aws_db_instance.postgres.address
    port     = 5432
    dbname   = "neuralretail"
    url      = "postgresql://neuralretail:${var.db_password}@${aws_db_instance.postgres.address}:5432/neuralretail"
  })
}

resource "aws_secretsmanager_secret" "redis_credentials" {
  name                    = "${local.name_prefix}/redis-credentials"
  description             = "NeuralRetail ElastiCache Redis connection"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "redis_credentials" {
  secret_id = aws_secretsmanager_secret.redis_credentials.id
  secret_string = jsonencode({
    host = aws_elasticache_replication_group.redis.primary_endpoint_address
    port = 6379
    url  = "rediss://${aws_elasticache_replication_group.redis.primary_endpoint_address}:6379"
  })
}

resource "aws_secretsmanager_secret" "app_secrets" {
  name                    = "${local.name_prefix}/app-secrets"
  description             = "NeuralRetail application secrets"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "app_secrets" {
  secret_id = aws_secretsmanager_secret.app_secrets.id
  secret_string = jsonencode({
    secret_key       = "CHANGE_ME_IN_CONSOLE"
    mlflow_s3_bucket = aws_s3_bucket.model_store.bucket
    openlineage_url  = ""
    airflow_password = "CHANGE_ME_IN_CONSOLE"
  })
}

# =============================================================================
# ECR -- Docker image repositories
# =============================================================================

resource "aws_ecr_repository" "fastapi" {
  name                 = "${local.name_prefix}/fastapi"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_ecr_repository" "streamlit" {
  name                 = "${local.name_prefix}/streamlit"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

# =============================================================================
# Outputs
# =============================================================================

output "eks_cluster_name"      { value = aws_eks_cluster.main.name }
output "eks_cluster_endpoint"  { value = aws_eks_cluster.main.endpoint }
output "rds_endpoint"          { value = aws_db_instance.postgres.address }
output "redis_endpoint"        { value = aws_elasticache_replication_group.redis.primary_endpoint_address }
output "s3_model_store_bucket" { value = aws_s3_bucket.model_store.bucket }
output "ecr_fastapi_url"       { value = aws_ecr_repository.fastapi.repository_url }
output "ecr_streamlit_url"     { value = aws_ecr_repository.streamlit.repository_url }
output "db_secret_arn"         { value = aws_secretsmanager_secret.db_credentials.arn }
output "redis_secret_arn"      { value = aws_secretsmanager_secret.redis_credentials.arn }
