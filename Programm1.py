# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 08:52:31 2026

@author: rodd1
"""

# Basic Code
print("Data Analysis")

grades = [80, 55, 90, 40, 70]
sum(grades)

average = sum(grades) / len(grades)
print(average)



# Temperature Code
days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
temperatures = [80, 55, 90, 40, 70]
highest = max(temperatures)
lowest = min(temperatures)
average = sum(temperatures) / len(temperatures)
highest_day = days[temperatures.index(highest)]
lowest_day = days[temperatures.index(lowest)]
print("Highest temperature:", highest, "on", highest_day)
print("Lowest temperature:", lowest, "on", lowest_day)
print("Average temperature:", average)




