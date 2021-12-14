---
Author: Elvyna Tunggawan
Created at: December 14, 2021
---

# Predicting E-Commerce Cart Abandonment

## Domain Background

In the past decade, e-commerce platforms have been rapidly growing. On a survey on Organisation for Economic Co-operation and Development (OECD) countries, it was found that more than half of the individuals made at least one online purchase within the last 12 months [^1]. Additionally, the Covid-19 pandemic resulted in a huge shift from brick-and-mortar business towards online shopping. In the UK, China, Germany, and the US, the share of retail sales from online purchases increased by 4-7% in 2020 compared to 2019 [^2]. As more people become more used to online purchases, there are increasing numbers of e-commerce platforms. To be a market leader, having a reliable platform and providing an amazing user experience become more important.

Although every e-commerce platform may require different purchasing steps, they mainly share a similar conversion funnel. To maximize their revenue, they strive to increase the overall conversion rate, i.e., aiming to convert as many web/app visitors into their customers.

![crazyegg-conversion-funnel](../img/crazyegg-conversion-funnel.png)
<p align='center'>Simple illustration of a conversion funnel in e-commerce platforms. Source: <a href="https://www.crazyegg.com/blog/ecommerce-conversion-funnel/">crazyegg.com</a></p>

[^1]: *Unlocking the Potential of E-commerce*. (2019, March). OECD. https://www.oecd.org/going-digital/unlocking-the-potential-of-e-commerce.pdf

[^2]: Alonso, V., Boar, C., Frost, J., Gambacorta, L., and Liu, J. (2021, January 12). *E-commerce in the pandemic and beyond*. Bank for International Settlements. https://www.bis.org/publ/bisbull36.pdf


> Student briefly details background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited. A discussion of the student's personal motivation for investigating a particular problem in the domain is encouraged but not required.

## Problem Statement

One of the latter parts of the conversion funnel is checking out the shopping cart (and purchasing the booking).

- Customers might abandon their cart or bookings
- User experience vs. potential loss from abandoned cart/bookings
- If we can predict whether the cart will be abandoned, the ecommerce platform may give different treatment to the users (e.g., offer coupon code to prevent abandonment, or offer upselling for users with low likelihood of abandoning their bookings)

> Student clearly describes the problem that is to be solved. The problem is well defined and has at least one relevant potential solution. Additionally, the problem is quantifiable, measurable, and replicable.

## Datasets

We use the dataset provided by [Data Mining Cup 2013](https://www.data-mining-cup.com/reviews/dmc-2013/)[^k], which contains 429,013 rows of e-commerce sessions with 24 columns. The following table describes the dataset structure.

[^k]: Data Mining Cup 2013. https://www.data-mining-cup.com/reviews/dmc-2013/

| Column name      | Description |
| ---------------- | ----------- |
| sessionNo        | running number of the session       |
| startHour        | hour in which the session has begun        |
| startWeekday        | day of week in which the session has begun (1: Mon, 2: Tue, ..., 7: Sun)        |
| duration        | time in seconds passed since start of the session        |
| cCount       | number of the products clicked on |
| cMinPrice    | lowest price of a product clicked on |
| cMaxPrice    | highest price of a product clicked on |
| cSumPrice    | sum of the prices of all products clicked on |
| bCount       | number of products put in the shopping basket |
| bMinPrice    | lowest price of all products put in the shopping basket |
| bMaxPrice    | highest price of all products put in the shopping basket |
| bSumPrice    | sum of theprices of all products put in the shopping basket|
| bStep    | purchase processing step (1,2,3,4,5)|
| onlineStatus    | indication whether the customer is online|
| availability    | delivery status|
| customerID    | customer ID|
| maxVal    | maximum admissible purchase price for the customer |
| customerScore    | customer evaluation from the point of view of the shop |
| accountLifetime    | lifetime of the customer's account in months |
| payments    | number of payments affected by the customer |
| age    | age of the customer |
| address    | form of address of the customer (1: Mr, 2: Mrs, 3: company)|
| lastOrder    | time in days passed since the last order|
| order    | outcome of the session (y: purchase, n: non-purchase) |

This dataset is stored as a text file (`transact_train.txt`) with pipe (`|`) as the delimiter values. We aim to create a model that predicts whether the order is purchased (`y`) or not (`n`).

> The dataset(s) and/or input(s) to be used in the project are thoroughly described. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included. It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

## Proposed Solution

A solution statement — the solution proposed for the problem given.

> Student clearly describes a solution to the problem. The solution is applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, the solution is quantifiable, measurable, and replicable.

## Benchmark Model

The provided dataset contains 67% of confirmed orders. As a simple benchmark, we aim to train a model with higher prediction accuracy than this. *Any relevant reserarch?*

A benchmark model — some simple or historical model or result to compare the defined solution to.

> A benchmark model is provided that relates to the domain, problem statement, and intended solution. Ideally, the student's benchmark model provides context for existing methods or known information in the domain and problem given, which can then be objectively compared to the student's solution. The benchmark model is clearly defined and measurable.

## Evaluation Metrics

A set of evaluation metrics — functional representations for how the solution can be measured.

> Student proposes at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model presented. The evaluation metric(s) proposed are appropriate given the context of the data, the problem statement, and the intended solution.

## Project Design

An outline of the project design — how the solution will be developed and results obtained.

> Student summarizes a theoretical workflow for approaching a solution given the problem. A discussion is made as to what strategies may be employed, what analysis of the data might be required, or which algorithms will be considered. The workflow and discussion provided align with the qualities of the project. Small visualizations, pseudocode, or diagrams are encouraged but not required.

Organisation:

> The proposal follows a well-organized structure and would be readily understood by its intended audience. Each section is written in a clear, concise and specific manner. Few grammatical and spelling mistakes are present. All resources used and referenced are properly cited.