# encoding=utf-8
from python.monkdata import monk1, monk1test, monk2, monk2test, monk3, monk3test, attributes
from python.dtree import entropy, averageGain, select, buildTree, check, allPruned
from python.drawtree_qt5 import drawTree
import statistics
import random
from pprint import PrettyPrinter

def partition(data, fraction):
    assert 0 < fraction < 1

    ldata = list(data)
    random.shuffle(ldata)
    breakpoint = int(len(data) * fraction)
    return ldata[:breakpoint], ldata[breakpoint:]


def assignment_1():
    print("*** ASSIGNMENT 1 ***")

    e_m1 = entropy(monk1test)
    e_m2 = entropy(monk2test)
    e_m3 = entropy(monk3test)

    print(
        """Entropies:
M1Test - {}
M2Test - {}
M3Test - {}

        """.format(e_m1, e_m2, e_m3)
    )

def assignment_3():
    print("*** ASSIGNMENT 3 ***")

    for dataset, name in {monk1: 'M1', monk2: 'M2', monk3: 'M3'}.items():
        attr_gain = []
        for attr in range(6):
            attr_gain.append(averageGain(dataset, attributes[attr]))

        attr_gain = [ round(e, 7) for e in attr_gain ]
        print("{}: {}".format(name, attr_gain))
    
    print("\n")

def assignment_5():

    print("*** ASSIGNMENT 5 ***")

    t_monk1 = buildTree(monk1, attributes)
    t_monk2 = buildTree(monk2, attributes)
    t_monk3 = buildTree(monk3, attributes)

    result_text = "{} -- E_train: {}; E_test: {}"

    print(result_text.format('MONK1', 1.0 - check(t_monk1, monk1), 1.0 - check(t_monk1, monk1test)))
    print(result_text.format('MONK2', 1.0 - check(t_monk2, monk2), 1.0 - check(t_monk2, monk2test)))
    print(result_text.format('MONK3', 1.0 - check(t_monk3, monk3), 1.0 - check(t_monk3, monk3test)))

    print("\n")


def optimum_prune(tree, val_data):

    def get_local_opt(tp):
        opt = (None, 0)
        for tree, perf in tp:
            opt = (tree, perf) if perf >= opt[1] else opt
        return opt

    optimum = (None, 0)
    while True:
        pruned_trees = allPruned(tree)
        performance = [ (t, check(t, val_data)) for t in pruned_trees ]
        local_opt = get_local_opt(performance)

        #print('Current optimum: {}, new optimum: {}'.format(optimum[1], local_opt[1]))

        if local_opt[1] > optimum[1]:
            optimum = local_opt
        else:
            break
    
    return optimum

def assignment_7():

    print("*** ASSIGNMENT 3 ***")
    
    samples = 10
    fractions = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    datasets = {
        'monk1' : monk1,
        'monk3' : monk3
    }

    results = {}

    for dataset_name, dataset in datasets.items(): 
        results[dataset_name] = {}

        for fraction in fractions:
            reliabilities = []
            for i in range(samples):
                train, validation = partition(dataset, fraction)
                tree = buildTree(train, attributes)
                opt_tree, reliability = optimum_prune(tree, validation)
                reliabilities.append(reliability)

            errors = list(map(lambda x: 1.0 - x, reliabilities))
            
            results[dataset_name][fraction] = {
                'mean': statistics.mean(errors),
                'median': statistics.median(errors),
                'std': statistics.stdev(errors),
                'max': max(errors),
                'min': min(errors)
            }

    pp = PrettyPrinter(indent=4)
    pp.pprint(results)

if __name__ == '__main__':
    assignment_1()
    assignment_3()
    assignment_5()
    assignment_7()
