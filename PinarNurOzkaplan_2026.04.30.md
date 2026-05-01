
---

## Student Name: Pınar Nur Özkaplan

## Project Title: Transfer Learning for Adaptive Collective Communication in Large Language Model Training

## Current Stage: Exploration

## Date Range: April 15 –  May 4 , 2026

---

## 1. Planned Goals for This Week

1. Transition from a policy-score based Agent GNN to an explicit routing prediction formulation.
2.  Improve the SimNet GNN with simulator-generated data for completion-time prediction


---

## 2. Actual Progress This Week

2.1 Reformulation of the Agent

- The initial agent was redesigned from a policy-score prediction setup to an explicit learning formulation for routing and scheduling.
- Routing is modeled as edge-level binary classification, while scheduling is modeled as a regression task.

2.2 Multi-head GNN Architecture
- Implemented a GATv2-based GNN with a multi-head output:
  - Routing head for edge selection
  - Scheduling head for predicting communication order
- Introduced masking so that scheduling loss is computed only on active edges.

2.3 Collective-aware Extension
- Extended the model to handle multiple collective operations (AllGather and AlltoAll).
- Added collective-type encoding to node features to enable conditional learning.

2.4 Results
- The model successfully learns routing and scheduling jointly across collectives.
Test performance (AllGather + AlltoAll):
- Routing Accuracy: 0.9917  
- Precision: 0.9834  
- Recall: ~1.000  
- Scheduling MAE: 0.0078  

---

## 3. Issue List This Week

### Issue 1

- Issue:
The model failed to generalize when trained on mixed collective operations.

- What has already been tried:
Included collective-type information as an input feature. Added a one-hot encoding of the collective type to node features.

- Result: Resolved the issue and restored performance.

### Issue 2


- Issue:
There is a design uncertainty regarding whether the agent output should be constrained to a tree structure or remain as a general routing graph, especially when handling multiple collective operations.

- What has already been tried:
The current model predicts routing decisions at the edge level without structural constraints, allowing flexible communication patterns.

- Result:
The model successfully learns different routing behaviors for different collectives, but does not enforce tree structures.

- Estimated possible solution:
Introduce a tree-based formulation (e.g., parent selection) as a constrained variant for tree-friendly collectives, while maintaining the general routing formulation for others.

- Final solution:
To be explored in the next stage.


---

## 4. Plan for Next Week

1. Reformulate the agent output to a tree-based (parent selection) representation.

---

## 5. Do You Need Support?

 No support needed.

---

## 6. One-Sentence Summary

A collective-aware multi-head Agent GNN was successfully developed to jointly learn routing and scheduling across different collective operations, while raising the question of whether a tree-constrained formulation should be adopted.

---

## 7. Self-Check Before Submission

- [x] I have clearly written the planned goals and actual progress for this week
- [x] I have listed all issues encountered this week
- [x] I have clearly written my plan for next week
- [x] I have indicated whether I need support
