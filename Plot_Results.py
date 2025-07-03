import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from itertools import cycle
import cv2 as cv
from sklearn.metrics import roc_curve


def plot_results():
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC', 'FOR', 'PT', 'CSI', "BA", 'FM', 'BM', 'MK', 'lrplus', 'lrminus', 'DOR', 'prevalence']
    Graph_Term = [4, 5, 6,  7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    Algorithm = ['TERMS', 'DO', 'WSO', 'DMO', 'DFA', 'Proposed']
    Classifier = ['TERMS', 'RNN', 'VGG16', 'CNN', 'MDARNN', 'PROPOSED']
    # value = eval[4, :, 4:]

    Batch_Size = ['50', '100', '150', '200', '250', '300', '350']
    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[1], eval.shape[0]))
        for k in range(eval.shape[1]):
            for l in range(eval.shape[0]):
                if j == 9:
                    Graph[k, l] = eval[l, k, Graph_Term[j] + 4]
                else:
                    Graph[k, l] = eval[l, k, Graph_Term[j] + 4]

        plt.plot(Batch_Size, Graph[0, :], color='y', linewidth=8, marker='o', markerfacecolor='blue', markersize=12,
                 label="TSO-FSRA-AEB7")
        plt.plot(Batch_Size, Graph[1, :], color='g', linewidth=8, marker='*', markerfacecolor='red', markersize=12,
                 label="BWO-FSRA-AEB7")
        plt.plot(Batch_Size, Graph[2, :], color='b', linewidth=8, marker='D', markerfacecolor='green', markersize=12,
                 label="CO-FSRA-AEB7")
        plt.plot(Batch_Size, Graph[3, :], color='m', linewidth=8, marker='>', markerfacecolor='yellow', markersize=12,
                 label="CFO-FSRA-AEB7")
        plt.plot(Batch_Size, Graph[4,:], color='k', linewidth=8, marker='^', markerfacecolor='cyan', markersize=12,
                 label="ACFO-FSRA-AEB7")
        plt.xticks(Batch_Size, ('50', '100', '150', '200', '250', '300', '350'), size=14)
        plt.xlabel('Epochs', size=14)
        plt.ylabel(Terms[Graph_Term[j]], size=14)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/Dataset_%s_line.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
        X = np.arange(7)
        ax.bar(X + 0.00, Graph[5, :], color=[0.5, 0.5, 0.9], width=0.15, label="CNN")
        ax.bar(X + 0.15, Graph[6, :], color='g', width=0.15, label="DCNN")
        ax.bar(X + 0.30, Graph[7, :], color='c', width=0.15, label="OAM-Net")
        ax.bar(X + 0.45, Graph[8, :], color='y', width=0.15, label="EfficientNetB7 ")
        ax.bar(X + 0.60, Graph[9, :], color='k', width=0.15, label="ACFO-FSRA-AEB7")
        plt.xticks(X + 0.30, ('50', '100', '150', '200', '250', '300', '350'), size=14)
        plt.xlabel('Epochs', size=14)
        plt.ylabel(Terms[Graph_Term[j]], size=14)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
                   ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/Dataset_%s_BarGraph.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()


def plot_tables():
    # matplotlib.use('TkAgg')
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC', 'FOR', 'PT', 'CSI', "BA", 'FM', 'BM', 'MK', 'lrplus', 'lrminus', 'DOR', 'prevalence']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    Algorithm = ['TERMS', 'TSO-FSRA-AEB7', 'BWO-FSRA-AEB7', 'CO-FSRA-AEB7', 'CFO-FSRA-AEB7', 'ACFO-FSRA-AEB7']
    Classifier = ['TERMS', 'CNN', 'DCNN', 'OAM-Net', 'EfficientNetB7', 'ACFO-FSRA-AEB7']
    Steps_per_Epoch = ['Adam', 'SGD', 'RMS Props', 'AdaGrad', 'AdaDelta']
    Feature = ['1st Fold', '2nd Fold', '3rd Fold', '4th Fold', '5th Fold']
    for m in range(eval.shape[0]-1):
        value = eval[m, :, 4:11]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms[0:7])
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, :])
        print('--------------------------------------------------', Feature[m], ' -Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms[0:7])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
        print('---------------------------------------------------', Feature[m], ' -Classifier Comparison',
              '--------------------------------------------------')
        print(Table)



def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_results_conv():
    Result = np.load('Fitness.npy', allow_pickle=True)
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'TSO-FSRA-AEB7', 'BWO-FSRA-AEB7', 'CO-FSRA-AEB7', 'CFO-FSRA-AEB7', 'ACFO-FSRA-AEB7']

    for i in range(Result.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = stats(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Dataset', i + 1, 'Statistical Report ',
              '--------------------------------------------------')

        print(Table)

        length = np.arange(50)
        Conv_Graph = Fitness[i]

        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=4, marker='*', markerfacecolor='red', markersize=12,
                 label='TSO-FSRA-AEB7')
        plt.plot(length, Conv_Graph[1, :], color='c', linewidth=4, marker='*', markerfacecolor='green',
                 markersize=12,
                 label='BWO-FSRA-AEB7')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=4, marker='*', markerfacecolor='cyan',
                 markersize=12,
                 label='CO-FSRA-AEB7')
        plt.plot(length, Conv_Graph[3, :], color='y', linewidth=4, marker='*', markerfacecolor='magenta',
                 markersize=12,
                 label='DCFO-FSRA-AEB7')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=4, marker='*', markerfacecolor='black',
                 markersize=12,
                 label='ACFO-FSRA-AEB7')
        plt.xlabel('Iteration', size=14)
        plt.ylabel('Cost Function', size=14)
        plt.legend(loc=1)
        plt.savefig("./Results/Dataset_%s_Conv.png" % (i + 1))
        plt.show()


def Plot_ROC():
    lw = 2
    cls = ['CNN', 'DCNN', 'OAM-Net', 'EfficientNetB7', 'ACFO-FSRA-AEB7']
    for a in range(1):
        Actual = np.load('Target.npy', allow_pickle=True).astype('int')
        colors = cycle(["blue", "r", "crimson", "gold", "black"])  # "cornflowerblue","darkorange", "aqua"
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i],
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", size=14)
        plt.ylabel("True Positive Rate", size=14)
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/Dataset_%s_ROC.png" % (a + 1)
        plt.savefig(path1)
        plt.show()

def Sample_Images():
    Orig = np.load('Images.npy', allow_pickle=True)
    Prep = np.load('Preprocessed_Images.npy', allow_pickle=True)
    ind = [10, 60, 110, 170, 220, 270, 330]
    # fig, ax = plt.subplots(2, 3)
    # plt.suptitle("Sample Images from Dataset")
    # plt.subplot(2, 3, 1)
    # plt.title('Image-1')
    # plt.imshow(Orig[ind[0]])
    # plt.subplot(2, 3, 2)
    # plt.title('Image-2')
    # plt.imshow(Orig[ind[1]])
    # plt.subplot(2, 3, 3)
    # plt.title('Image-3')
    # plt.imshow(Orig[ind[2]])
    # plt.subplot(2, 3, 4)
    # plt.title('Image-4')
    # plt.imshow(Orig[ind[3]])
    # plt.subplot(2, 3, 5)
    # plt.title('Image-5')
    # plt.imshow(Orig[ind[4]])
    # plt.subplot(2, 3, 6)
    # plt.title('Image-6')
    # plt.imshow(Orig[ind[5]])
    # plt.show()
    cv.imwrite('./Results/Sample Images/Original-img-' + str(0 + 1) + '.png', Orig[ind[0]])
    cv.imwrite('./Results/Sample Images/Original-img-' + str(1 + 1) + '.png', Orig[ind[1]])
    cv.imwrite('./Results/Sample Images/Original-img-' + str(2 + 1) + '.png', Orig[ind[2]])
    cv.imwrite('./Results/Sample Images/Original-img-' + str(3 + 1) + '.png', Orig[ind[3]])
    cv.imwrite('./Results/Sample Images/Original-img-' + str(4 + 1) + '.png', Orig[ind[4]])
    cv.imwrite('./Results/Sample Images/Original-img-' + str(5 + 1) + '.png', Orig[ind[5]])
    cv.imwrite('./Results/Sample Images/Original-img-' + str(6 + 1) + '.png', Orig[ind[6]])
    cv.imwrite('./Results/Sample Images/Preprocessed-img-' + str(0 + 1) + '.png', Prep[ind[0]])
    cv.imwrite('./Results/Sample Images/Preprocessed-img-' + str(1 + 1) + '.png', Prep[ind[1]])
    cv.imwrite('./Results/Sample Images/Preprocessed-img-' + str(2 + 1) + '.png', Prep[ind[2]])
    cv.imwrite('./Results/Sample Images/Preprocessed-img-' + str(3 + 1) + '.png', Prep[ind[3]])
    cv.imwrite('./Results/Sample Images/Preprocessed-img-' + str(4 + 1) + '.png', Prep[ind[4]])
    cv.imwrite('./Results/Sample Images/Preprocessed-img-' + str(5 + 1) + '.png', Prep[ind[5]])
    cv.imwrite('./Results/Sample Images/Preprocessed-img-' + str(6 + 1) + '.png', Prep[ind[6]])



if __name__ == '__main__':
    plot_results()
    plot_tables()
    Plot_ROC()
    plot_results_conv()
    # Sample_Images()