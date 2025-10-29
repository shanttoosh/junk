# Reinforcement Learning (RL)

## Table of Contents
1. Problem Formulation (MDP)
2. Policy, Value, and Optimality
3. Exploration vs Exploitation
4. Q-Learning and Policy Gradient (Conceptual)
5. Applications and Considerations
6. Interview Insights

---

## 1) Problem Formulation (MDP)

Markov Decision Process (MDP): (S, A, P, R, γ)
- States (S), Actions (A)
- Transition dynamics P(s'|s,a)
- Reward R(s,a)
- Discount γ ∈ [0,1]

Goal: find a policy π(a|s) maximizing expected discounted return.

---

## 2) Policy, Value, and Optimality

- Policy π: mapping from states to action probabilities
- State-value Vπ(s): expected return from s following π
- Action-value Qπ(s,a): expected return from s taking a then following π
- Optimal policy π*: maximizes value for all states; satisfies Bellman optimality

---

## 3) Exploration vs Exploitation

- Exploitation: choose best-known action (greedy)
- Exploration: try actions to gather information
- ε-greedy: with probability ε explore, else exploit
- Softmax/boltzmann: stochastic choices weighted by estimated values

Balance is critical; insufficient exploration leads to suboptimal policies.

---

## 4) Q-Learning and Policy Gradient (Conceptual)

Q-Learning (value-based): learn Q*(s,a) without a model of P
- Update rule drives estimates toward Bellman target using observed rewards and next-state estimates
- Off-policy: learns greedy policy while exploring with ε-greedy

Policy Gradient (actor-based): directly optimize policy parameters
- Maximize expected return via gradient ascent
- Stochastic policies enable differentiability
- Variance reduction via baselines (advantage functions)

Deep RL: use neural networks to approximate value/policy for high-dimensional states; requires careful stabilization (target networks, replay buffers).

---

## 5) Applications and Considerations

Applications:
- Robotics, games (AlphaGo), recommender systems (slates/bandits), ad bidding

Considerations:
- Sample efficiency: RL often data-hungry
- Safety: exploration may be risky; use offline RL or constraints
- Reward design: mis-specified rewards lead to unintended behaviors
- Non-stationarity: environment drift requires continual learning

---

## 6) Interview Insights

- RL vs supervised: feedback is delayed, sparse; data is collected via interaction
- Bandits vs full RL: bandits have no state; simpler exploration/exploitation trade-offs
- Business angle: Optimizes sequential decisions (offers, recommendations, operations) under uncertainty
