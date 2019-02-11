# You need to learn a function with n inputs.
# For given number of inputs, we will generate random function.
# Your task is to learn it
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import solutionmanager as sm

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SolutionModel, self).__init__()
        assert output_size == 1
        self.model = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, x):
        return self.model.forward(x)

    def calc_loss(self, output, target):
        return self.loss_fn(output, target)

    def calc_predict(self, output):
        return output.round()


class Solution():
    def __init__(self):
        self = self

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        # init model
        for p in model.parameters():
            nn.init.uniform_(p, -1.0, +1.0)
        model.train()
        loss_fn = torch.nn.BCELoss()
        optimizer = optim.LBFGS(model.parameters(), lr=0.01)
        while True:
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1:
                break

            output = model(train_data)
            total = train_target.view(-1).size(0)
            predicted = output.round()
            correct = predicted.eq(train_target.view_as(predicted)).long().sum().item()
            if correct == total:
                break

            def closure():
                optimizer.zero_grad()
                output = model(train_data)
                loss = loss_fn(output, train_target)
                loss.backward()
                return loss
            optimizer.step(closure)
            step += 1

        return step

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 10000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, input_size, seed):
        random.seed(seed)
        data_size = 1 << input_size
        data = torch.FloatTensor(data_size, input_size)
        target = torch.FloatTensor(data_size)
        for i in range(data_size):
            for j in range(input_size):
                input_bit = (i>>j)&1
                data[i,j] = float(input_bit)
            target[i] = float(random.randint(0, 1))
        return (data, target.view(-1, 1))

    def create_case_data(self, case):
        input_size = min(3+case, 7)
        data, target = self.create_data(input_size, case)
        return sm.CaseData(case, Limits(), (data, target), (data, target)).set_description("{} inputs".format(input_size))


class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=-1)
