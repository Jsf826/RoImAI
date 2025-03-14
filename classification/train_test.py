import torch
from torch.nn.modules.activation import Softmax
from tqdm import tqdm

from utils import *
import copy
import time
from sklearn.metrics import confusion_matrix, average_precision_score


def mul_train(epoches, net, trainloader, testloader, optimizer, scheduler, lr_adjt, dataset, CELoss, tree, device,
              devices,
              save_name):
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    max_val_acc = 0
    best_epoch = 0
    if len(devices) > 1:
        ids = list(map(int, devices))
        netp = torch.nn.DataParallel(net, device_ids=ids)
    for epoch in range(epoches):
        epoch_start = time.time()
        # print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0

        order_correct = 0
        family_correct = 0
        species_correct_soft = 0
        species_correct_sig = 0

        order_total = 0
        family_total = 0
        species_total = 0

        idx = 0
        if lr_adjt == 'Cos':
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, epoches, lr[nlr])

        for batch_idx, (inputs1, inputs2, targets) in tqdm(enumerate(trainloader), total=len(trainloader),
                                                           desc='Epoch {}/{}'.format(epoch + 1, epoches)):
            idx = batch_idx

            inputs1, inputs2, targets = inputs1.to(device), inputs2.to(device), targets.to(device)
            # 修改
            order_targets, family_targets, target_list_sig = get_order_family_target(targets, device, dataset)

            optimizer.zero_grad()

            # if len(devices) > 1:
            #     xc1_sig, xc2_sig, xc3, xc3_sig = netp(inputs)
            # else:
            # xc1_sig, xc2_sig, xc3, xc3_sig = net(inputs)
            xc1_sig, xc2, xc2_sig = net(inputs1, inputs2)

            tree_loss = tree(torch.cat([xc1_sig, xc2_sig], 1), target_list_sig, device)
            if dataset == 'CUB':
                leaf_labels = torch.nonzero(targets > 50, as_tuple=False)
            elif dataset == 'Air':
                leaf_labels = torch.nonzero(targets > 99, as_tuple=False)
            elif dataset == 'SSMG' or dataset == 'Mul_SSMG':
                leaf_labels = torch.nonzero(targets > 2, as_tuple=False)
            if leaf_labels.shape[0] > 0:
                if dataset == 'CUB':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 51
                elif dataset == 'Air':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 100
                elif dataset == 'SSMG' or dataset == 'Mul_SSMG':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 3
                select_fc_soft = torch.index_select(xc2, 0, leaf_labels.squeeze())
                ce_loss_species = CELoss(select_fc_soft.to(torch.float64), select_leaf_labels)
                loss = ce_loss_species + tree_loss
            else:
                loss = tree_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():

                _, family_predicted = torch.max(xc1_sig.data, 1)
                family_total += family_targets.size(0)
                family_correct += family_predicted.eq(family_targets.data).cpu().sum().item()

                if leaf_labels.shape[0] > 0:
                    select_xc2 = torch.index_select(xc2, 0, leaf_labels.squeeze())
                    select_xc2_sig = torch.index_select(xc2_sig, 0, leaf_labels.squeeze())
                    _, species_predicted_soft = torch.max(select_xc2.data, 1)
                    _, species_predicted_sig = torch.max(select_xc2_sig.data, 1)
                    species_total += select_leaf_labels.size(0)
                    species_correct_soft += species_predicted_soft.eq(select_leaf_labels.data).cpu().sum().item()
                    species_correct_sig += species_predicted_sig.eq(select_leaf_labels.data).cpu().sum().item()

        if lr_adjt == 'Step':
            scheduler.step()

        # train_order_acc = 100. * order_correct / order_total
        train_family_acc = 100. * family_correct / family_total
        train_species_acc_soft = 100. * species_correct_soft / species_total
        train_species_acc_sig = 100. * species_correct_sig / species_total
        train_loss = train_loss / (idx + 1)
        epoch_end = time.time()

        print(
            'Iteration %d, train_family_acc = %.5f,train_species_acc_soft = %.5f,train_species_acc_sig = %.5f, train_loss = %.6f, Time = %.1fs' % \
            (epoch, train_family_acc, train_species_acc_soft, train_species_acc_sig, train_loss,
             (epoch_end - epoch_start)))

        test_family_acc, test_species_acc_soft, test_species_acc_sig, test_loss = mul_test(net, testloader, CELoss,
                                                                                           tree,
                                                                                           device, dataset)

        if test_species_acc_soft > max_val_acc:
            max_val_acc = test_species_acc_soft
            best_epoch = epoch
            net.cpu()
            torch.save(net, 'checkpoints/' + dataset + '/model_' + save_name + '.pth')
            net.to(device)

    print('\n\nBest Epoch: %d, Best Results: %.5f' % (best_epoch, max_val_acc))


def mul_test(net, testloader, CELoss, tree, device, dataset):
    epoch_start = time.time()
    with torch.no_grad():
        net.eval()
        test_loss = 0

        order_correct = 0
        family_correct = 0
        species_correct_soft = 0
        species_correct_sig = 0

        order_total = 0
        family_total = 0
        species_total = 0

        idx = 0

        for batch_idx, (inputs1, inputs2, targets) in tqdm(enumerate(testloader), total=len(testloader), desc='test:'):
            idx = batch_idx

            inputs1, inputs2, targets = inputs1.to(device), inputs2.to(device), targets.to(device)
            # 修改
            order_targets, family_targets, target_list_sig = get_order_family_target(targets, device, dataset)

            # optimizer.zero_grad()

            xc1_sig, xc2, xc2_sig = net(inputs1, inputs2)
            tree_loss = tree(torch.cat([xc1_sig, xc2_sig], 1), target_list_sig, device)

            if dataset == 'CUB':
                leaf_labels = torch.nonzero(targets > 50, as_tuple=False)
            elif dataset == 'Air':
                leaf_labels = torch.nonzero(targets > 99, as_tuple=False)
            elif dataset == 'SSMG' or dataset == 'Mul_SSMG':
                leaf_labels = torch.nonzero(targets > 2, as_tuple=False)
            if leaf_labels.shape[0] > 0:
                if dataset == 'CUB':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 51
                elif dataset == 'Air':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 100
                elif dataset == 'SSMG' or dataset == 'Mul_SSMG':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 3
                select_fc_soft = torch.index_select(xc2, 0, leaf_labels.squeeze())
                ce_loss_species = CELoss(select_fc_soft.to(torch.float64), select_leaf_labels)
                loss = ce_loss_species + tree_loss
            else:
                loss = tree_loss

            test_loss += loss.item()

            _, family_predicted = torch.max(xc1_sig.data, 1)
            family_total += family_targets.size(0)
            family_correct += family_predicted.eq(family_targets.data).cpu().sum().item()

            if leaf_labels.shape[0] > 0:
                select_xc2 = torch.index_select(xc2, 0, leaf_labels.squeeze())
                select_xc2_sig = torch.index_select(xc2_sig, 0, leaf_labels.squeeze())
                _, species_predicted_soft = torch.max(select_xc2.data, 1)
                _, species_predicted_sig = torch.max(select_xc2_sig.data, 1)
                species_total += select_leaf_labels.size(0)
                species_correct_soft += species_predicted_soft.eq(select_leaf_labels.data).cpu().sum().item()
                species_correct_sig += species_predicted_sig.eq(select_leaf_labels.data).cpu().sum().item()


        test_family_acc = 100. * family_correct / family_total
        test_species_acc_soft = 100. * species_correct_soft / species_total
        test_species_acc_sig = 100. * species_correct_sig / species_total
        test_loss = test_loss / (idx + 1)
        epoch_end = time.time()
        print(
            'test_family_acc = %.5f,test_species_acc_soft = %.5f,test_species_acc_sig = %.5f, test_loss = %.6f, Time = %.4s' % \
            (test_family_acc, test_species_acc_soft, test_species_acc_sig, test_loss,
             epoch_end - epoch_start))

    return test_family_acc, test_species_acc_soft, test_species_acc_sig, test_loss


def train(epoches, net, trainloader, testloader, optimizer, scheduler, lr_adjt, dataset, CELoss, tree, device, devices,
          save_name):
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    max_val_acc = 0
    best_epoch = 0
    if len(devices) > 1:
        ids = list(map(int, devices))
        netp = torch.nn.DataParallel(net, device_ids=ids)
    for epoch in range(epoches):
        epoch_start = time.time()
        # print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0

        order_correct = 0
        family_correct = 0
        species_correct_soft = 0
        species_correct_sig = 0

        order_total = 0
        family_total = 0
        species_total = 0

        idx = 0
        if lr_adjt == 'Cos':
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, epoches, lr[nlr])

        for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), total=len(trainloader),
                                                 desc='Epoch {}/{}'.format(epoch + 1, epoches)):
            idx = batch_idx

            inputs, targets = inputs.to(device), targets.to(device)
            # 修改
            order_targets, family_targets, target_list_sig = get_order_family_target(targets, device, dataset)

            optimizer.zero_grad()

            if len(devices) > 1:
                xc1_sig, xc2_sig, xc3, xc3_sig = netp(inputs)
            else:
                # xc1_sig, xc2_sig, xc3, xc3_sig = net(inputs)
                xc1_sig, xc2, xc2_sig = net(inputs)
            tree_loss = tree(torch.cat([xc1_sig, xc2_sig], 1), target_list_sig, device)
            if dataset == 'CUB':
                leaf_labels = torch.nonzero(targets > 50, as_tuple=False)
            elif dataset == 'Air':
                leaf_labels = torch.nonzero(targets > 99, as_tuple=False)
            elif dataset == 'SSMG' or dataset == 'Mul_SSMG':
                leaf_labels = torch.nonzero(targets > 2, as_tuple=False)
            if leaf_labels.shape[0] > 0:
                if dataset == 'CUB':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 51
                elif dataset == 'Air':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 100
                elif dataset == 'SSMG' or dataset == 'Mul_SSMG':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 3
                select_fc_soft = torch.index_select(xc2, 0, leaf_labels.squeeze())
                ce_loss_species = CELoss(select_fc_soft.to(torch.float64), select_leaf_labels)
                loss = ce_loss_species + tree_loss
            else:
                loss = tree_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():


                _, family_predicted = torch.max(xc1_sig.data, 1)
                family_total += family_targets.size(0)
                family_correct += family_predicted.eq(family_targets.data).cpu().sum().item()

                if leaf_labels.shape[0] > 0:
                    select_xc2 = torch.index_select(xc2, 0, leaf_labels.squeeze())
                    select_xc2_sig = torch.index_select(xc2_sig, 0, leaf_labels.squeeze())
                    _, species_predicted_soft = torch.max(select_xc2.data, 1)
                    _, species_predicted_sig = torch.max(select_xc2_sig.data, 1)
                    species_total += select_leaf_labels.size(0)
                    species_correct_soft += species_predicted_soft.eq(select_leaf_labels.data).cpu().sum().item()
                    species_correct_sig += species_predicted_sig.eq(select_leaf_labels.data).cpu().sum().item()

        if lr_adjt == 'Step':
            scheduler.step()

        # train_order_acc = 100. * order_correct / order_total
        train_family_acc = 100. * family_correct / family_total
        train_species_acc_soft = 100. * species_correct_soft / species_total
        train_species_acc_sig = 100. * species_correct_sig / species_total
        train_loss = train_loss / (idx + 1)
        epoch_end = time.time()
        # print(
        #     'Iteration %d, train_order_acc = %.5f,train_family_acc = %.5f,train_species_acc_soft = %.5f,train_species_acc_sig = %.5f, train_loss = %.6f, Time = %.1fs' % \
        #     (epoch, train_order_acc, train_family_acc, train_species_acc_soft, train_species_acc_sig, train_loss,
        #      (epoch_end - epoch_start)))
        print(
            'Iteration %d, train_family_acc = %.5f,train_species_acc_soft = %.5f,train_species_acc_sig = %.5f, train_loss = %.6f, Time = %.1fs' % \
            (epoch, train_family_acc, train_species_acc_soft, train_species_acc_sig, train_loss,
             (epoch_end - epoch_start)))

        test_family_acc, test_species_acc_soft, test_species_acc_sig, test_loss = test(net, testloader, CELoss, tree,
                                                                                       device, dataset)

        if test_species_acc_soft > max_val_acc:
            max_val_acc = test_species_acc_soft
            best_epoch = epoch
            net.cpu()
            torch.save(net, 'checkpoints/' + dataset + '/model_' + save_name + '.pth')
            net.to(device)

    print('\n\nBest Epoch: %d, Best Results: %.5f' % (best_epoch, max_val_acc))


def test(net, testloader, CELoss, tree, device, dataset):
    epoch_start = time.time()
    with torch.no_grad():
        net.eval()
        test_loss = 0

        order_correct = 0
        family_correct = 0
        species_correct_soft = 0
        species_correct_sig = 0

        order_total = 0
        family_total = 0
        species_total = 0

        idx = 0

        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader), total=len(testloader), desc='test:'):
            idx = batch_idx

            inputs, targets = inputs.to(device), targets.to(device)
            # 修改
            order_targets, family_targets, target_list_sig = get_order_family_target(targets, device, dataset)

            # optimizer.zero_grad()

            xc1_sig, xc2, xc2_sig = net(inputs)
            tree_loss = tree(torch.cat([xc1_sig, xc2_sig], 1), target_list_sig, device)

            if dataset == 'CUB':
                leaf_labels = torch.nonzero(targets > 50, as_tuple=False)
            elif dataset == 'Air':
                leaf_labels = torch.nonzero(targets > 99, as_tuple=False)
            elif dataset == 'SSMG' or dataset == 'Mul_SSMG':
                leaf_labels = torch.nonzero(targets > 2, as_tuple=False)
            if leaf_labels.shape[0] > 0:
                if dataset == 'CUB':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 51
                elif dataset == 'Air':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 100
                elif dataset == 'SSMG' or dataset == 'Mul_SSMG':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 3
                select_fc_soft = torch.index_select(xc2, 0, leaf_labels.squeeze())
                ce_loss_species = CELoss(select_fc_soft.to(torch.float64), select_leaf_labels)
                loss = ce_loss_species + tree_loss
            else:
                loss = tree_loss

            test_loss += loss.item()

            _, family_predicted = torch.max(xc1_sig.data, 1)
            family_total += family_targets.size(0)
            family_correct += family_predicted.eq(family_targets.data).cpu().sum().item()

            if leaf_labels.shape[0] > 0:
                select_xc2 = torch.index_select(xc2, 0, leaf_labels.squeeze())
                select_xc2_sig = torch.index_select(xc2_sig, 0, leaf_labels.squeeze())
                _, species_predicted_soft = torch.max(select_xc2.data, 1)
                _, species_predicted_sig = torch.max(select_xc2_sig.data, 1)
                species_total += select_leaf_labels.size(0)
                species_correct_soft += species_predicted_soft.eq(select_leaf_labels.data).cpu().sum().item()
                species_correct_sig += species_predicted_sig.eq(select_leaf_labels.data).cpu().sum().item()



        test_family_acc = 100. * family_correct / family_total
        test_species_acc_soft = 100. * species_correct_soft / species_total
        test_species_acc_sig = 100. * species_correct_sig / species_total
        test_loss = test_loss / (idx + 1)
        epoch_end = time.time()
        print(
            'test_family_acc = %.5f,test_species_acc_soft = %.5f,test_species_acc_sig = %.5f, test_loss = %.6f, Time = %.4s' % \
            (test_family_acc, test_species_acc_soft, test_species_acc_sig, test_loss,
             epoch_end - epoch_start))

    return test_family_acc, test_species_acc_soft, test_species_acc_sig, test_loss



def test_AP(model, dataset, test_set, test_data_loader, device):
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        model.eval()
        for j, (images, labels) in enumerate(test_data_loader):
            images = images.to(device)
            labels = labels.to(device)
            select_labels = labels[:, test_set.to_eval]
            if dataset == 'CUB' or dataset == 'Air':
                y_order_sig, y_family_sig, y_species_sof, y_species_sig = model(images)
                batch_pMargin = torch.cat([y_order_sig, y_family_sig, torch.softmax(y_species_sof, dim=1)], dim=1).data
            else:
                y_order_sig, y_species_sof, y_species_sig = model(images)
                batch_pMargin = torch.cat([y_order_sig, torch.softmax(y_species_sof, dim=1)], dim=1).data

            predicted = batch_pMargin > 0.5
            total += select_labels.size(0) * select_labels.size(1)
            correct += (predicted.to(torch.float64) == select_labels).sum()
            cpu_batch_pMargin = batch_pMargin.to('cpu')
            y = select_labels.to('cpu')
            if j == 0:
                test = cpu_batch_pMargin
                test_y = y
            else:
                test = torch.cat((test, cpu_batch_pMargin), dim=0)
                test_y = torch.cat((test_y, y), dim=0)
        score = average_precision_score(test_y, test, average='micro')
        print('Accuracy:' + str(float(correct) / float(total)))
        print('Precision score:' + str(score))
