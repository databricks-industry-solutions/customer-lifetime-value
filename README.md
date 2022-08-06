##Calculating the Probability of Future Customer Engagement

In non-subscription retail models, customers come and go with no long-term commitments, making it very difficult to determine whether a customer will return in the future. Determining the probability that a customer will re-engage is critical to the design of effective marketing campaigns. Different messaging and promotions may be required to incentivize customers who have likely dropped out to return to our stores. Engaged customers may be more responsive to marketing that encourages them to expand the breadth and scale of purchases with us. Understanding where our customers land with regard to the probability of future engagement is critical to tailoring our marketing efforts to them.  

The *Buy 'til You Die* (BTYD) models popularized by Peter Fader and others leverage two basic customer metrics, *i.e.* the recency of a customer's last engagement and the frequency of repeat transactions over a customer's lifetime, to derive a probability of future re-engagement. This is done by fitting customer history to curves describing the distribution of purchase frequencies and engagement drop-off following a prior purchase. The math behind these models is fairly complex but thankfully it's been encapsulated in the [lifetimes](https://pypi.org/project/Lifetimes/) library, making it much easier for traditional enterprises to employ. The purpose of this notebook is to examine how these models may be applied to customer transaction history and how they may be deployed for integration in marketing processes.

___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

