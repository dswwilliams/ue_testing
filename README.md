
What's the objective for the ue_testing repo? 

- It needs to define a model
- Then pass the model and options to instantiate a Tester class
- Then we just need to call methods in the tester class

- so the main issue is defining the options and the model

- these should be stripped down versions of the training ones
- i.e. the model class should have as few methods as possible
- there should be as few options as possible



- we want to be able to define any model in a class and test it, i.e.:
    1. we need to init the network


- Need to implement ensembles and mcd at the very least as models
- 