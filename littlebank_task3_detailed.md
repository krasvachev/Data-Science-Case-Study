## Introduction

The table represents the relation between the features of the data and their importance for the GLM ElasticNet model. Positive values mean that the feature increases the probability of classifying the sample as 1, i.e. a higher probability for a customer to subscribe to the product. Higher values lead to higher impact for the classification. For example, outcome_previous.success is 0.2101, i.e. it impacts the ML model to classify the sample as a successful outcome. It also has a bigger impact on the classification than the low_temp feature. To the contrary, negative values impact the model to classify the sample as a person who won’t subscribe. The higher the negative value, the stronger the impact of the feature. For example, num_employed leads the model towards classifying the sample as non-subscription.

## A Closer Look

### Analysis of the Data

* If the previous campaign was successful, it is very likely that the client would subscribe again to the product.

* If the contact was made in March, the chances for successful subscription are high. For July, there is some chance of a positive outcome of the marketing. For November, there is a low chance of a positive outcome. If the call was in May, most certainly the customer won’t subscribe to the product.

* People who are retired or in full-time education are more likely to purchase the product than people working in the industry. However, people working in the industry have more financial opportunities than people who are retired or have a job in the educational system.

* When the number of employees is high, the subscriptions for the product decrease. The same situation could be observed if the employment variation increases.

* It could be said with high probability that being on a credit product does not influence the customer's decision to purchase the product. And that is not good news.

* The probability of a successful conversion rate is approx. 8.3% (0.083). That means around 10% of the people will subscribe to the savings account after the marketing. These results are also affirmed by the implemented ML models.

* There isn’t any visible improvement in the subscriptions, even with the advertisement campaign. This means that there is a problem with the campaign. The conclusion is derived from the applied analysis and models.

### Recommendation

Advertisement of the product is important, but the quality of the product is more important. The most important part in a business, however, is the customers. A product should solve the needs of a customer, remove pains, give efficiency. 

* The advertisement campaign is disastrous. With or without it, the relative percentage of subscriptions is almost the same. The campaign should be more focused on the right customer segment, communicated with an accurate message via the most appropriate medium (Social media, Website, not via mobile or landline).

* Find the true needs of the customers and personalise the product for them. An improvement of the product will be needed in order to increase the conversion rate. 

* There is no connection between being a client of the bank and purchasing the product. Feedback from the clients should be taken to see if they like the products of the bank and what would make them eager to purchase the product. The right relationship is: if someone is a satisfied client of the bank, there is a higher probability of purchasing the product of the bank or recommending it to another person.

* Target different working groups more precisely and more efficiently. (How many people have landlines nowadays? How many people accept offers from a cold call?)
