"""
FILE_NAME:Multitasking Optimization
Author:shaolong wei、hongcheng yao
Contact:yaohongcheng18@gmail.com
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

df = pd.read_csv('dataset/RankedData/RankedNottingham.csv', header=None)
max_features = df.shape[1] - 1
global_feature_selection_counts = np.zeros(max_features, dtype=int)
global_data_splits = {}


def load_task_data(task_path):
    data = pd.read_csv(task_path, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    if task_path not in global_data_splits:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        global_data_splits[task_path] = (X_train, X_test, y_train, y_test)
    return global_data_splits[task_path]


def initialize_wolves(num_wolves, num_features):
    return np.random.randint(2, size=(num_wolves, num_features))


def initialize_wolves_based_on_global_info(num_wolves, selected_features_status):
    global global_feature_selection_counts
    task_global_counts = global_feature_selection_counts[selected_features_status == 1]

    if np.sum(task_global_counts) > 0:
        selection_probability = task_global_counts / np.sum(task_global_counts)
    else:
        selection_probability = np.ones(sum(selected_features_status == 1)) / sum(selected_features_status == 1)

    wolves = np.zeros((num_wolves, sum(selected_features_status == 1)), dtype=int)
    for i in range(num_wolves):
        wolves[i] = np.random.rand(sum(selected_features_status == 1)) < selection_probability

    return wolves


def evaluate_fitness(wolves, X_train, X_test, y_train, y_test):
    fitness_scores = []
    for wolf in wolves:
        features_selected = wolf[:X_train.shape[1]] == 1
        if not any(features_selected):
            fitness_scores.append(0)
            continue
        X_selected_train = X_train[:, features_selected]
        X_selected_test = X_test[:, features_selected]
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_selected_train, y_train)
        predictions = knn.predict(X_selected_test)
        accuracy = accuracy_score(y_test, predictions)

        penalty = sum(features_selected) / X_train.shape[1]
        fitness_score = 0.9999 * accuracy - 0.0001 * penalty
        fitness_scores.append(fitness_score)
    return np.array(fitness_scores)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def update_wolf_position(alpha_pos, beta_pos, delta_pos, wolf_pos, a):
    A = 2 * a * np.random.rand() - a
    C = 2 * np.random.rand()
    D_alpha = np.abs(C * alpha_pos - wolf_pos)
    D_beta = np.abs(C * beta_pos - wolf_pos)
    D_delta = np.abs(C * delta_pos - wolf_pos)
    X1 = alpha_pos - A * D_alpha
    X2 = beta_pos - A * D_beta
    X3 = delta_pos - A * D_delta
    new_pos = (X1 + X2 + X3) / 3
    return np.where(np.random.rand(*new_pos.shape) < sigmoid(new_pos), 1, 0)


def select_leading_wolves(fitness_scores):
    sorted_indices = np.argsort(fitness_scores)[::-1]
    return sorted_indices[:3]


def map_selection_to_original(selected_features_status, feature_selection_result):
    original_selection_result = np.zeros_like(selected_features_status)
    considered_features_indices = np.where(selected_features_status == 1)[0]
    for i, selected in enumerate(feature_selection_result):
        if i < len(considered_features_indices):
            original_index = considered_features_indices[i]
            original_selection_result[original_index] = selected
    return original_selection_result


def gwo(num_wolves, num_features, task_path, selected_features_status, max_iter=300):
    X_train, X_test, y_train, y_test = load_task_data(task_path)

    wolves = initialize_wolves_based_on_global_info(num_wolves, selected_features_status)

    fitness_scores = evaluate_fitness(wolves, X_train, X_test, y_train, y_test)

    leading_indices = select_leading_wolves(fitness_scores)
    alpha_wolf, beta_wolf, delta_wolf = wolves[leading_indices[0]], wolves[leading_indices[1]], wolves[
        leading_indices[2]]

    for iteration in range(max_iter):
        a = 2 - iteration * (2 / max_iter)

        for i, wolf in enumerate(wolves):
            wolves[i] = update_wolf_position(alpha_wolf, beta_wolf, delta_wolf, wolf, a)

        fitness_scores = evaluate_fitness(wolves, X_train, X_test, y_train, y_test)
        leading_indices = select_leading_wolves(fitness_scores)
        alpha_wolf, beta_wolf, delta_wolf = wolves[leading_indices[0]], wolves[leading_indices[1]], wolves[
            leading_indices[2]]

        print(f"Iteration {iteration + 1}: Best fitness = {fitness_scores[leading_indices[0]]}")

    return wolves[leading_indices[0]], fitness_scores[leading_indices[0]]


def run_gwo_for_multiple_tasks_with_global_info(task_paths):
    global global_feature_selection_counts
    results = {}
    for i, task_path in enumerate(task_paths, 1):
        X_train, X_test, y_train, y_test = load_task_data(task_path)
        num_features = X_train.shape[1]

        selected_features_status_path = f'dataset/generateTask/Nottingham/Task{i}_selected_features.csv'
        selected_features_status = pd.read_csv(selected_features_status_path, header=None).values.flatten()

        best_features, best_fitness = gwo(20, num_features, task_path, selected_features_status, 100)
        original_selection_result = map_selection_to_original(selected_features_status, best_features)
        global_feature_selection_counts += original_selection_result

        results[f"Task{i}"] = (best_features, best_fitness)

        selected_features_indices = np.where(best_features == 1)[0]
        res_file_path = f'dataset/TaskRes/Nottinghamglobal/svm/Task{i}_res.csv'
        with open(res_file_path, 'w') as f:
            for index in selected_features_indices:
                f.write(f'{index}\n')
        print(f"Task {i} best features saved to {res_file_path}")

    return results


# 执行多任务GWO
task_paths = [f"dataset/generateTask/Nottingham/Task{i}.csv" for i in range(1, 9)]
results = run_gwo_for_multiple_tasks_with_global_info(task_paths)

# 输出结果
for task, (features, fitness) in results.items():
    print(f"{task}: Best Fitness = {fitness}, Features = {features}")

for a in global_data_splits:
    print(a)


def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    bac = (sen + spe) / 2
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    return acc, sen, spe, bac, ppv, npv


def process_task(task_id, task_path):
    X_train, X_test, y_train, y_test = global_data_splits[task_path]

    classifiers = {
        "SVM": SVC(),
    }

    index_filename = f"dataset/TaskRes/Nottinghamglobal/svm/Task{task_id}_res.csv"
    index_data = pd.read_csv(index_filename, header=None)
    indexes = index_data.iloc[:, 0].values

    metrics_record = {name: {"ACC": [], "SEN": [], "SPE": [], "BAC": [], "PPV": [], "NPV": []} for name in
                      classifiers}

    for _ in range(5):
        X_train_selected = X_train[:, indexes]
        X_test_selected = X_test[:, indexes]

        for name, clf in classifiers.items():
            clf.fit(X_train_selected, y_train)
            y_pred = clf.predict(X_test_selected)
            acc, sen, spe, bac, ppv, npv = calculate_metrics(y_test, y_pred)
            metrics_record[name]["ACC"].append(acc)
            metrics_record[name]["SEN"].append(sen)
            metrics_record[name]["SPE"].append(spe)
            metrics_record[name]["BAC"].append(bac)
            metrics_record[name]["PPV"].append(ppv)
            metrics_record[name]["NPV"].append(npv)

    statistics = {}
    for name, metrics in metrics_record.items():
        stats = {}
        for metric_name, values in metrics.items():
            max_value = max(values)
            min_value = min(values)
            avg_value = np.mean(values)
            stats[metric_name] = {"Max": max_value, "Min": min_value, "Avg": avg_value}
        statistics[name] = stats
    return statistics


all_tasks_statistics = {}

for task_id, task_path in enumerate(task_paths, start=1):
    statistics = process_task(task_id, task_path)
    all_tasks_statistics[f"Task {task_id}"] = statistics
    print(f"Task {task_id} Results:")
    for classifier, stats in statistics.items():
        for metric, sub_stats in stats.items():
            print(
                f"{classifier} - {metric} - Max: {sub_stats['Max']:.4f}, Min: {sub_stats['Min']:.4f}, Avg: {sub_stats['Avg']:.4f}")

best_tasks_per_classifier = {}

for classifier in ["SVM"]:
    max_avg_acc = 0
    best_task = None
    for task, stats in all_tasks_statistics.items():
        if stats[classifier]["ACC"]["Avg"] > max_avg_acc:
            max_avg_acc = stats[classifier]["ACC"]["Avg"]
            best_task = task
    best_tasks_per_classifier[classifier] = (best_task, max_avg_acc)

print("Best task per classifier based on average accuracy:")
for classifier, (task, avg_accuracy) in best_tasks_per_classifier.items():
    print(f"{classifier}: {task} with Avg Accuracy: {avg_accuracy:.4f}")

# 单独输出每个分类器平均精度最大的任务的统计数据
for classifier, (task, _) in best_tasks_per_classifier.items():
    print(f"\nDetailed Statistics for {classifier} - Best Performing Task: {task}")
    for metric, sub_stats in all_tasks_statistics[task][classifier].items():
        print(f"{metric} - Max: {sub_stats['Max']:.4f}, Min: {sub_stats['Min']:.4f}, Avg: {sub_stats['Avg']:.4f}")
