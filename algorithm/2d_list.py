# -*- coding: utf-8 -*-
import numpy as np

if __name__ == '__main__':
    course_list = ['core C++', 'coreJava', 'Servlet', 'JSP', 'EJB']
    student_num = 20
    course_num = len(course_list)

    score_array = np.random.randint(0, 100, 100).reshape(20, 5)
    print(score_array)
    for i in range(len(score_array)):
        print('score of student %d:' % i)
        print(score_array[i])
        print('the total score of student %d:' % i)
        print(score_array.sum(axis=1))

    for i in range(score_array.shape[1]):
        print('the average score of %s:' % course_list[i])
        print(score_array.mean(axis=0)[i])
