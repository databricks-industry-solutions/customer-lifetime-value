## Calculating the Probability of Future Customer Engagement

In non-subscription retail models, customers come and go with no long-term commitments, making it very difficult to determine whether a customer will return in the future. In addition, customers frequently settle into a pattern of regular spend with retailers with whom they maintain a long-term relationship.  But occasionally, customers will spend at higher rates before returning back to their previous norm.  Both of these patterns make effective projections of customer spending very challenging for most retail organizations.

The *Buy 'til You Die* (BTYD) models popularized by Peter Fader and others leverage a few basic customer metrics, *i.e.* the recency of a customer's last engagement, the frequency of repeat transactions over a customer's lifetime, the average monetary spend associated with those transactions, and the length (term) of a customer's time engaged with a retailer to derive probabilistic estimations of both a customer's future spend and that customer's likelihood to remain engaged.  Using these values, we can project likely future spend, a value we frequently refer to as the customer's lifetime value (CLV).

The math behind this approach is fairly complex but thankfully it's been encapsulated in the [btyd](https://pypi.org/project/btyd/) library, making it much easier for traditional enterprises to employ. The purpose of this notebook is to examine how these models may be applied to customer transaction history to estimate CLV.

In this notebook, we are going to create two models that are used to estimate lifetime value.  The first of these will be used to estimate the probability of customer retention through a certain point in time.  The second will be used to calculate the estimated monetary value through that same point in time.  Together, these estimates can be combined to calculate a customer's value through and extended period of time.
___

&copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

To run this accelerator, clone this repo into a Databricks workspace. Attach the RUNME notebook to any cluster and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. Execute the multi-step-job to see how the pipeline runs.

The job configuration is written in the RUNME notebook in json format. The cost associated with running the accelerator is the user's responsibility.
