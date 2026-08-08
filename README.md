# Federated Learning based Privacy-Preserving Social Networks

<p align="center">

  <h3 align="center">Privacy-Preserving Federated Graph Neural Network for Social Network Analysis</h3>

  <p align="center">
    A fairness-aware and privacy-preserving Federated Graph Neural Network framework for decentralized social network analysis.
  </p>

</p>

---

## 📌 Overview

Social networks generate large amounts of graph-structured data representing relationships, interactions, and communities. Traditional Graph Neural Network (GNN) approaches generally require centralized access to this data, which can expose sensitive user information.

This project proposes a **Federated Learning based Privacy-Preserving Social Network Analysis framework** that combines:

- Federated Learning
- Graph Neural Networks (GNNs)
- Local Differential Privacy (LDP)
- Secure Subgraph Aggregation (SecureSA)
- Secure Node Augmentation (SecureNA)
- Fairness-aware aggregation
- Trust and robust aggregation mechanisms

The key idea is to allow multiple clients to collaboratively train a global graph model **without sharing their raw graph data**.

---

## 🎯 Objectives

The major objectives of the project are:

1. Develop a Federated Learning framework for decentralized graph data.
2. Apply Graph Neural Networks to learn node representations.
3. Protect sensitive information using Local Differential Privacy.
4. Implement Secure Subgraph Aggregation (SecureSA).
5. Implement Secure Node Augmentation (SecureNA).
6. Reduce bias using fairness-aware aggregation.
7. Improve robustness against unreliable or malicious client updates.
8. Evaluate the framework based on accuracy, fairness, privacy, and communication efficiency.

---

## 🏗️ System Architecture

The system follows an end-to-end federated learning pipeline:

```text
                 ┌──────────────────────────┐
                 │      Global Server       │
                 │                          │
                 │  Federated Aggregation   │
                 │  Fairness Evaluation    │
                 │  Privacy Accounting     │
                 └────────────┬─────────────┘
                              │
                    Global Model Broadcast
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │   Client 1  │     │   Client 2  │ ... │   Client N  │
   │             │     │             │     │             │
   │ Local Graph │     │ Local Graph │     │ Local Graph │
   │    GCN/GAT  │     │    GCN/GAT  │     │    GCN/GAT  │
   │             │     │             │     │             │
   │  SecureSA   │     │  SecureSA   │     │  SecureSA   │
   │  SecureNA   │     │  SecureNA   │     │  SecureNA   │
   │     LDP     │     │     LDP      │     │     LDP     │
   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
          │                   │                   │
          └──────────── Secure Updates ───────────┘
                              │
                              ▼
                       Global Aggregation
