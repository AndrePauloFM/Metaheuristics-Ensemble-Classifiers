# -*- coding: utf-8 -*-
from sklearn import datasets
import inquirer
import evaluateMetaheuristicas as evaluate

"""# DataBase Selection"""
questions = [
  inquirer.List('base',
                message="Select DataBase:",
                choices=['Digits', 'Wine', 'Breast Cancer'],
            ),
]
answers = inquirer.prompt(questions)

def functionDigits():
    print("DataBase Digits Selected")
    dataBase = datasets.load_digits()
    return dataBase

def functionWine():
    print("DataBase Wine Selected")
    dataBase = datasets.load_wine()
    return dataBase

def functionBreast():
    print("DataBase Breast Cancer Selected")
    dataBase = datasets.load_breast_cancer()
    return dataBase

def default():
    print("Select one Database:")

if __name__ == "__main__":
    switch = {
        "Digits": functionDigits,
        "Wine": functionWine,
        "Breast Cancer": functionBreast
    }

    case = switch.get(answers["base"], default)
    dataBase = case()
    evaluate.evaluateMetaheuristicas(dataBase)
