# Product–Outlet Targeting

My understanding is that we need to develop a solution that automates which products should be targeted for outlets.

## 1. Business understanding and objectives

The very first thing that we need to take into consideration is: what is the ultimate goal? What are we trying to optimize? We need to choose target products that optimize traffic, profit, revenue? In general, I would try to understand the business problem and add some KPIs that could provide a comprehensive picture of the problem. Shall we optimize the commercial value? Do we need to make sure that we increase availability rate for each product? Also, some products could be important strategically (new launches).

Also, I would like to ask some more generic information about: How will the model output be used? How often shall we change the targets (shall we predict targets for the next day or the next week)? Define some KPIs such as expected demand, expected margin, execution cost, added value of targeting.


| Topic                | Questions / KPIs to consider                                               |
| -------------------- | -------------------------------------------------------------------------- |
| Ultimate objective   | Traffic, profit, revenue, commercial value, availability rate              |
| Strategic priorities | Important products, including new launches                                 |
| Model usage          | How the model output will be used                                          |
| Targeting frequency  | Next day, next week, or another time horizon                               |
| KPIs                 | Expected demand, expected margin, execution cost, added value of targeting |


## 2. Understand the existing targeting approach

We definitely need to know the existing approach and how well this approach performs with respect to those KPIs. How do we decide targets? Have we done any study? We may need to collect some metrics (understand why some products are not targets for some outlets).

## 3. Data requirements and initial analysis

So the next step would be to run some statistics to check the demand of products by outlet, to see the value of products by outlet (profit, revenue), the cost for products by outlet (how costly it is to bring a product to a specific outlet).

I would request sales volume, price, and margin by product and outlet, complete product catalog, outlet capacity, product attributes, consumer demand and local demographics.


| Information                                       | Purpose                                         |
| ------------------------------------------------- | ----------------------------------------------- |
| Sales volume, price, margin by product and outlet | Demand and value of products by outlet          |
| Cost by product and outlet                        | Cost of bringing a product to a specific outlet |
| Complete product catalog and product attributes   | Product-level information                       |
| Outlet capacity                                   | Feasibility and operational constraints         |
| Consumer demand and local demographics            | Local demand information                        |


## 4. Candidate selection and modelling

Then I would continue with the intervention design: identify valid / feasible pairings. Then I could find a way to model the problem:

**Added value for x-y targeting (risk × profit).**

A first approach could be estimating expected value from historical data; the next one could be product-outlet ranking models (build product features and outlet features and compute the matching score).

Again, after scoring the candidates, we need to make sure that we take into consideration supply, outlet capacity, budget and other constraints. So we need to try to solve an optimization problem under constraints (if too complicated we could start with greedy experimentation).

## 5. Proposed end-to-end solution

Therefore, I would develop a model that has the following components:

### 5.1 Data ingestion

Outlet master data, product catalog, sales, historical targets, operational constraints.

### 5.2 Feature pipelines and model inference

Produce features available at decision time, candidate eligibility, and opportunity estimates.

### 5.3 Assignment optimization

Apply budgets, outlet constraints, strategic priorities, and supply availability.

### 5.4 Business review and execution

Sales teams approve or override recommendations; actions enter existing workflows, evaluate the solution thoroughly.

### 5.5 Outcome measurement and feedback

Track execution, availability, incremental sales, cost, and reasons for rejected targets.

## 6. Technical considerations

Some technical considerations include: scalability, data quality, cold start, prevent information leakage, monitor performance, explainability.

## 7. Evaluation and rollout

In evaluation, we shall consider offline evaluation using a number of metrics that give a comprehensive picture of how our solution performs. We could try shadow deployment (without changing the existing decision) and gradual integration, starting from a small targeted group and scaling up.

## 8. Summary

In summary, I would start with business understanding, data inspection, proposed solution, production integration and validation and rollout.