## Update Logs

### Nov. 2020: Update to the version 1.0.5

- Change the methods to get the models, using the fully register methods
- Change the methods to get loss functions. For the custom loss classes, we directly register them in the registry. And for the `PyTorch` loss classes, we wrap them into a new class in order to realize the registry mechanism

- Re-design the `coinfig/defaults.yaml`, add more flexibility to this file and make some setting more reasonable
- Add the `PyPi` install method

### Jun. 2020: The repo is open!!

