import pandas as pd
import numpy as np
import torch
from metrics import get_metrics
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import itertools
import os
import pprint

# static constants
HYPERPARAMS = ['learning_rate', 'num_iters', 'n_h', 'n_h_adv', 'dropout_rate', 'alpha']
intermediate_metrics = False

class Model(object):
    def __init__(self, params):
        self.params = params
        self.method = self.params['method']
        self.adversarial = self.method != 'basic'
        self.num_classes = self.params['num_classes']
        self.logpath = self.params['logpath']
        self.hyperparams = self.params['hyperparams']
        self.model = self.build_model()
        self.data = self.process_data()

    def valid_hyperparam(self, i):
        return (i < 3 or i == 4 or self.adversarial)

    def get_indexes(self):
        num_models = []
        for i in range(len(HYPERPARAMS)):
            if self.valid_hyperparam(i):
                num_models.append(range(len(self.hyperparams[HYPERPARAMS[i]])))
            else:
                num_models.append([None]) # placeholder value if no such hyperparameter
        return itertools.product(*num_models)

    def get_hyperparams(self, indexes):
        hyperparams = []
        for i in range(len(indexes)):
            if self.valid_hyperparam(i):
                hyperparams.append(self.hyperparams[HYPERPARAMS[i]][indexes[i]])
            else:
                hyperparams.append(None)
        return hyperparams

    def hyperparams_to_string(self, indexes):
        res = ''
        for i in range(len(HYPERPARAMS)):
            if i > 0:
                res += '-'
            if self.valid_hyperparam(i):
                res += HYPERPARAMS[i] + '_' + str(self.hyperparams[HYPERPARAMS[i]][indexes[i]])
        return res

    def build_model(self):
        models = {}
        for indexes in self.get_indexes():
                models[indexes] = self.build_single_model(indexes)
        return models

    def build_single_model(self, indexes):
        model = dict()
        m, n = len(self.params['Xtrain']), len(self.params['Xtrain'][0])
        m_valid, n_valid = len(self.params['Xvalid']), len(self.params['Xvalid'][0])
        m_test, n_test = len(self.params['Xtest']), len(self.params['Xtest'][0])
        n_h = self.hyperparams['n_h'][indexes[2]]

        model['model'] = torch.nn.Sequential(
            torch.nn.Linear(n, n_h),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.hyperparams['dropout_rate'][indexes[4]]),
            torch.nn.Linear(n_h, 1),
            torch.nn.Sigmoid(),
        )
        model['loss_fn'] = torch.nn.BCELoss(size_average=True)
        model['optimizer'] = torch.optim.Adam(model['model'].parameters(), lr=self.hyperparams['learning_rate'][indexes[0]])

        if self.adversarial:
            n_h_adv = self.hyperparams['n_h_adv'][indexes[3]]
            if self.num_classes > 2:
                n_h_out = self.num_classes
            else:
                n_h_out = 1

            if self.method == 'parity':
                n_adv = 1
            elif self.method == 'odds' or 'opportunity':
                n_adv = 2
            else:
                raise Exception('Unknown method: {}'.format(self.method))

            model['adv_model'] = torch.nn.Sequential(
                torch.nn.Linear(n_adv, n_h_adv),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.hyperparams['dropout_rate'][indexes[4]]),
	            torch.nn.Linear(n_h_adv, n_h_out),
                torch.nn.Sigmoid(),
            )
            if (self.num_classes > 2):
                model['adv_loss_fn'] = torch.nn.CrossEntropyLoss(size_average=True)
            else:
                model['adv_loss_fn'] = torch.nn.BCELoss(size_average=True)
            model['adv_optimizer'] = torch.optim.Adam(model['adv_model'].parameters(), lr=self.hyperparams['learning_rate'][indexes[0]])

        return model

    def process_data(self):
        data = dict()
        m, n = len(self.params['Xtrain']), len(self.params['Xtrain'][0])
        m_valid, n_valid = len(self.params['Xvalid']), len(self.params['Xvalid'][0])
        m_test, n_test = len(self.params['Xtest']), len(self.params['Xtest'][0])
        n_h = self.hyperparams['n_h']

        if self.method == 'opportunity':
            data['adv_train_mask'] = self.params['ytrain'] == 1
            data['adv_train_mask'] = torch.ByteTensor(data['adv_train_mask'].astype(int).values.reshape(m, 1))

            data['adv_valid_mask'] = self.params['yvalid'] == 1
            data['adv_valid_mask'] = torch.ByteTensor(data['adv_valid_mask'].astype(int).values.reshape(m_valid, 1))

            data['adv_test_mask'] = self.params['ytest'] == 1
            data['adv_test_mask'] = torch.ByteTensor(data['adv_test_mask'].astype(int).values.reshape(m_test, 1))

        data['Xtrain'] = Variable(torch.Tensor(self.params['Xtrain']))
        data['ytrain'] = Variable(torch.Tensor(np.expand_dims(np.asarray(self.params['ytrain']), 1)))
        data['Xvalid'] = Variable(torch.Tensor(self.params['Xvalid']))
        data['yvalid'] = Variable(torch.Tensor(np.expand_dims(np.asarray(self.params['yvalid']), 1)))
        data['Xtest'] = Variable(torch.Tensor(self.params['Xtest']))
        data['ytest'] = Variable(torch.Tensor(np.expand_dims(np.asarray(self.params['ytest']), 1)))

        data['ztrain'] = Variable(torch.Tensor(self.params['ztrain']).long())
        data['zvalid'] = Variable(torch.Tensor(self.params['zvalid']).long())
        data['ztest'] = Variable(torch.Tensor(self.params['ztest']).long())

        return data

    def train(self):
        for indexes in self.get_indexes():
            self.train_single_model(indexes)

    def load_trained_models(self):
        for indexes in self.get_indexes():
            hyperparam_values = self.hyperparams_to_string(indexes)
            modelfile = self.logpath + '-model/' + hyperparam_values + '-model.pth'
            self.model[indexes]['model'] = torch.load(modelfile)

    def create_dir(self, dirname):
        if (not os.path.exists(dirname)):
            os.makedirs(dirname)

    def train_single_model(self, indexes):
        # Load in model and data
        model = self.model[indexes]['model']
        loss_fn = self.model[indexes]['loss_fn']
        optimizer = self.model[indexes]['optimizer']
        Xtrain = self.data['Xtrain']
        Xvalid = self.data['Xvalid']
        Xtest = self.data['Xtest']
        ytrain = self.data['ytrain']
        yvalid = self.data['yvalid']
        ytest = self.data['ytest']
        ztrain = self.data['ztrain']
        zvalid = self.data['zvalid']
        ztest = self.data['ztest']
        if self.adversarial:
            adv_model = self.model[indexes]['adv_model']
            adv_loss_fn = self.model[indexes]['adv_loss_fn']
            adv_optimizer = self.model[indexes]['adv_optimizer']

        model.train()

        # Set up logging
        self.create_dir(self.logpath + '-training/')
        self.create_dir(self.logpath + '-metrics/')
        self.create_dir(self.logpath + '-model/')
        if self.adversarial:
            self.create_dir(self.logpath + '-adv/')
        hyperparam_values = self.hyperparams_to_string(indexes)
        logfile = self.logpath + '-training/' + hyperparam_values
        metrics_file = self.logpath + '-metrics/' + hyperparam_values + '-metrics.csv'
        metrics = []
        modelfile = self.logpath + '-model/' + hyperparam_values + '-model.pth'
        if self.adversarial:
            advfile = self.logpath + '-adv/' + hyperparam_values + '-adv.pth'
        writer = SummaryWriter(logfile)

        for t in range(self.hyperparams['num_iters'][indexes[1]]):
            # Forward step
            ypred_train = model(Xtrain)
            loss_train = loss_fn(ypred_train, ytrain)

            if self.adversarial:
                if self.method == 'parity':
                    adv_input_train = ypred_train
                elif self.method == 'odds':
                    adv_input_train = torch.cat((ypred_train, ytrain), 1)
                elif self.method == 'opportunity':
                    adv_input_train = torch.stack((torch.masked_select(ypred_train, self.data['adv_train_mask']),
                                                 torch.masked_select(ytrain, self.data['adv_train_mask'])), 1)
                    ztrain = torch.masked_select(ztrain, self.data['adv_train_mask'])

                zpred_train = adv_model(adv_input_train)
                adv_loss_train = adv_loss_fn(zpred_train, ztrain)

                combined_loss_train = loss_train - self.hyperparams['alpha'][indexes[5]] * adv_loss_train

            # Train log
            if t % 100 == 0:
                print('Iteration: {}'.format(t))

            # Save model
            if t > 0 and t % 1000 == 0:
                print(t)
                torch.save(model, modelfile)
                if self.adversarial:
                    torch.save(adv_model, advfile)

            # Backward step
            if self.adversarial:
                # adv update
                adv_optimizer.zero_grad()
                adv_loss_train.backward(retain_graph=True)
                adv_optimizer.step()
                # pred update
                optimizer.zero_grad()
                combined_loss_train.backward()
            else:
                optimizer.zero_grad()
                loss_train.backward()

            optimizer.step()

        # save final model
        torch.save(model, modelfile)
        if self.adversarial:
            torch.save(adv_model, advfile)
        writer.close()

        ypred_test = model(Xtest)
        zpred_test = None
        if self.adversarial:
            if self.method == 'parity':
                adv_input_test = ypred_test
            elif self.method == 'odds':
                adv_input_test = torch.cat((ypred_test, ytest), 1)
            elif self.method == 'opportunity':
                adv_input_test = torch.stack((torch.masked_select(ypred_test, self.data['adv_test_mask']),
                                             torch.masked_select(ytest, self.data['adv_test_mask'])), 1)
            zpred_test = adv_model(adv_input_test)

        print('Hii =================> Final test metrics for model with ' + self.hyperparams_to_string(indexes) + ' on test:')
        metrics_test = get_metrics(ypred_test.data.numpy(), ytest.data.numpy(), ztest.data.numpy(),  self.get_hyperparams(indexes), self.num_classes, 0, 'test_set')
        print(metrics_test)


    def eval(self):
        evalfile = self.logpath + '-eval.csv'
        test_metrics = []
        for indexes in self.get_indexes():
            test_metrics.append(self.eval_single_model(indexes))

        pd.concat(test_metrics).to_csv(evalfile)

    def eval_single_model(self, indexes):
        model = self.model[indexes]['model']
        # loss_fn = self.model[indexes]['loss_fn']
        # optimizer = self.model[indexes]['optimizer']
        Xtrain = self.data['Xtrain']
        Xvalid = self.data['Xvalid']
        Xtest = self.data['Xtest']
        ytrain = self.data['ytrain']
        yvalid = self.data['yvalid']
        ytest = self.data['ytest']
        ztrain = self.data['ztrain']
        zvalid = self.data['zvalid']
        ztest = self.data['ztest']

        model.eval()
        ypred_valid = model(Xvalid)
        zpred_valid = None
        if self.adversarial:
            adv_model = self.model[indexes]['adv_model']
            adv_model.eval()

            if self.method == 'parity':
                adv_input_valid = ypred_valid
                zpred_valid = adv_model(adv_input_valid)
            elif self.method == 'odds':
                adv_input_valid = torch.cat((ypred_valid, yvalid), 1)
                zpred_valid = adv_model(adv_input_valid)
            elif self.method == 'opportunity':
                zpred_valid = None

        if zpred_valid is not None:
            metrics_valid = pd.DataFrame(get_metrics(ypred_valid.data.numpy(), yvalid.data.numpy(), zvalid.data.numpy(), self.get_hyperparams(indexes), k=self.num_classes, evaluation_file='valid_set', zpred=zpred_valid.data.numpy()), index=[0])
        else:
            metrics_valid = pd.DataFrame(get_metrics(ypred_valid.data.numpy(), yvalid.data.numpy(), zvalid.data.numpy(), self.get_hyperparams(indexes), k=self.num_classes, evaluation_file='valid_set'), index=[0])
        print
        print('Final test metrics for model with ' + self.hyperparams_to_string(indexes) + ' on validation:')
        pprint.pprint(metrics_valid)

        ypred_test = model(Xtest)
        zpred_test = None
        if self.adversarial:
            if self.method == 'parity':
                adv_input_test = ypred_test
                zpred_test = adv_model(adv_input_test)
            elif self.method == 'odds':
                adv_input_test = torch.cat((ypred_test, ytest), 1)
                zpred_test = adv_model(adv_input_test)
            elif self.method == 'opportunity':
                zpred_test = None

        if zpred_test is not None:
            metrics_test = pd.DataFrame(get_metrics(ypred_test.data.numpy(), ytest.data.numpy(), ztest.data.numpy(), self.get_hyperparams(indexes), k=self.num_classes, evaluation_file='test_set', zpred=zpred_test.data.numpy()), index=[0])
        else:
            metrics_test = pd.DataFrame(get_metrics(ypred_test.data.numpy(), ytest.data.numpy(), ztest.data.numpy(), self.get_hyperparams(indexes), k=self.num_classes, evaluation_file='test_set'), index=[0])
        print
        print('Final test metrics for model with ' + self.hyperparams_to_string(indexes) + ' on test:')
        pprint.pprint(metrics_test)
        return pd.concat([metrics_valid, metrics_test])



def write_log(writer, key, loss, iter):
    writer.add_scalar(key, loss.item(), iter)


def write_log_array(writer, key, array, iter):
    writer.add_text(key, np.array_str(array), iter)
