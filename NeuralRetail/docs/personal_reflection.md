# Personal Reflection & Post-Mortem

**Author:** Amdox Data Science Intern  
**Project:** NeuralRetail Enterprise AI Platform  
**Date:** April 2026  

---

## 1. Key Learnings & Achievements

Building NeuralRetail from the ground up as an end-to-end MLOps platform was an incredibly rewarding challenge. Moving beyond Jupyter notebooks and focusing on production-grade engineering principles fundamentally shifted my perspective on applied AI.

**Key Achievements:**
* **End-to-End MLOps:** Successfully implemented a complete lifecycle, from Kafka/Parquet data ingestion to Feast feature stores, model registry via MLflow, and automated retraining pipelines triggered by Evidently AI drift detection.
* **Champion-Challenger Automation:** Built a robust Kubernetes shadow deployment pattern utilizing Istio VirtualServices. This ensures new challenger models are tested against live traffic silently, guaranteeing zero-downtime and risk-free model promotion.
* **Production Infrastructure:** Wrote comprehensive Terraform IaC to provision AWS EKS, RDS, ElastiCache, and S3, paired with a fully hardened GitHub Actions CI/CD pipeline.
* **SLO Enforcement:** Implemented real Prometheus middleware to track P95 latency and integrated alerting rules to guarantee the < 500ms response time requirement.

## 2. Technical Challenges Overcome

* **Feature Store Serialization Issues:** Upgrading Feast to a production-grade Redis online store introduced serialization conflicts. I had to deep-dive into Feast's documentation and upgrade the `entity_key_serialization_version` to version 3, ensuring seamless materialization from the Silver layer.
* **Drift Detection API Migration:** Evidently AI recently shifted to a new `metric_v2` API in version 0.7.x. My initial implementation relied on deprecated `Report` modules. I successfully rewrote the `DriftMonitor` class to parse the new `Snapshot.dict()` format, extracting exact column-level K-S and Wasserstein metrics.
* **Airflow Webhook Connectivity:** Triggering Airflow DAGs externally required managing authentication and network routing carefully. Resolving connection refused errors involved ensuring the Airflow REST API was exposed correctly on the host network during local testing.

## 3. Future Roadmap (If given 3 more months)

If I had three additional months to expand NeuralRetail, I would focus on:

1. **Large Language Model (LLM) Integration:** Introduce a generative AI "Copilot" into the Streamlit dashboard using LangChain and an open-source model (like Llama 3). This would allow executives to ask natural language questions like *"Why did churn increase in the electronics category this week?"* and receive SHAP-backed narrative explanations.
2. **Reinforcement Learning for Pricing:** Currently, price elasticity is modelled using DoWhy causal inference. I would explore deep reinforcement learning (e.g., Proximal Policy Optimization) to dynamically suggest optimal pricing strategies that maximize long-term GMV rather than just short-term demand.
3. **Advanced Anomaly Detection:** Implement isolated isolation forests or autoencoders at the data ingestion layer to detect and quarantine fraudulent transactions or extreme data anomalies before they pollute the bronze data lake.
4. **Multi-Region Kubernetes Federation:** Scale the application deployment across multiple AWS regions for high availability and disaster recovery, utilizing ArgoCD for multi-cluster GitOps synchronization.

## 4. Conclusion

NeuralRetail stands as a comprehensive testament to modern AI engineering. It bridges the gap between predictive accuracy and operational reliability, delivering measurable business value through a resilient, scalable, and observable architecture. I am incredibly proud of the foundation built here.
