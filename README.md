# newyork


# run by: python kod.py
# goal
predict value of tip for the driver
# constraints
tip only recorded if customer paid with credit card, <br>
thus only using rides paid with credit card
# what we have learned
value of tip is not really connected with pickup/dropoff location <br>
best results with trip_distance, passanger_count, tolls, total_price <br>
tax would be good feature for classficator model for this issue <br>
# functions:
drop_non_creditcard() - drops rides not paid by credit card
data_for_linear- linear regression with only one feature
data_for_linear_multifeatures - using multiple features
# meaning of the graph
graph with black dots as target value <br>
red dots as value predicted
# output:
mean and absolut square error for training data <br>
mean and absolut square error for testing data <br>
variance score

