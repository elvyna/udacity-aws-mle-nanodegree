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

One of the earlier steps of the conversion funnel is adding items into the shopping cart. Although having the cart filled is one step closer to the purchase, there are possibilities that the users will abandon their cart. For example, the users might consider a shopping cart as a helper to save the items that they're interested in. This misuse might happen since not all e-commerce platforms have a wishlist or bookmark feature. 

Some e-commerce platforms might consider protecting the inventory that has been added to a shopping cart to ensure a seamless user experience, i.e., the items will be reserved by the users although they leave the cart for hours before purchasing them. However, this method is likely to end up with huge potential losses, especially if most users frequently abandon their carts.

## Problem Statement

Instead of protecting the inventory, e-commerce platforms usually introduce nudge marketing to influence user behaviour. However, nudges might backfire if they are implemented without understanding the user behaviour [^3].

[^3]: Sanghi, R., Gupta, S., Mishra, R.C., Singh, A., Abrol, H., Madan, J., & Sagar, M. (2019). *Nudge Marketing in E-Commerce Businesses*. International Journal of Science and Research (IJSR). https://www.ijsr.net/archive/v8i8/ART2020223.pdf

Analyzing users' purchasing intent has become one of the research areas in e-commerce. Most e-commerce platforms have pivoted to more personalized features and services, which is also the case for Pinterest [^4]. Website clickstream and user session data are the main data sources to understand the purchasing intent since they contain the whole picture of the e-commerce platform (not only the bookings) [^5] [^6]. If we could utilize the user activity logs to identify whether a cart is likely to be abandoned, the e-commerce platforms could provide more personalized treatment to the users. For example, they might offer coupon codes to convert the abandoning users into customers and offer some upselling for users with a high likelihood to purchase.

[^4]: Lo, C., Frankowski, D., & Leskovec, J. (2016). Understanding behaviors that lead to purchasing. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. https://doi.org/10.1145/2939672.2939729

[^5]: Sakar, C. O., Polat, S. O., Katircioglu, M., & Kastro, Y. (2018). Real-time prediction of online shoppers’ purchasing intention using multilayer perceptron and LSTM recurrent neural networks. *Neural Computing and Applications*, 31(10), 6893-6908. https://doi.org/10.1007/s00521-018-3523-0

[^6]: Kompan, M., Kassak, O., & Bielikova, M. (2019). The Short-term User Modeling for Predictive Applications. *Journal on Data Semantics, 8(1)*, 21–37. https://doi.org/10.1007/s13740-018-0095-1

## Datasets

In this project, we will use the dataset provided by [Data Mining Cup 2013](https://www.data-mining-cup.com/reviews/dmc-2013/)[^k], which contains 429,013 rows of e-commerce sessions with 24 columns. The following table describes the dataset structure.

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

## Proposed Solution

A solution statement — the solution proposed for the problem given.

> Student clearly describes a solution to the problem. The solution is applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, the solution is quantifiable, measurable, and replicable.

## Benchmark Model

The provided dataset contains 67% of confirmed orders. As a simple benchmark, we aim to train a model with higher prediction accuracy than this. *Any relevant research?*

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