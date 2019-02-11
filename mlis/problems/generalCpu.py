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

SIZE = 40
LAYERS = 6

class Solution():
    def __init__(self):
        self = self

    def create_model(self, input_size, output_size):
        assert output_size == 1
        lst = []
        for i in range(LAYERS):
            if i != 0:
                lst.append(nn.Tanh())
            lst.append(nn.Linear(input_size if i == 0 else SIZE,
                                 output_size if i == LAYERS-1 else SIZE))
        lst.append(nn.Sigmoid())
        return nn.Sequential(*lst)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        # init model
        torch.manual_seed(5)
        for p in model.parameters():
            nn.init.uniform_(p, -1.0, +1.0)
        model.train()
        loss_fn = torch.nn.BCELoss()
        optimizer = optim.RMSprop(model.parameters(), lr=0.00175751062)
        while True:
            time_left = context.get_timer().get_time_left()
            if time_left < 0.1:
                break

            optimizer.zero_grad()
            output = model(train_data)
            loss = loss_fn(output, train_target)
            loss.backward()
            total = train_target.view(-1).size(0)
            predicted = output.round()
            correct = predicted.eq(train_target.view_as(predicted)).long().sum().item()
            if correct == total:
                break

            optimizer.step()
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
