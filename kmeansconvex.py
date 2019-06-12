import numpy as np
import matplotlib.pyplot as plt
import math

flag = 0

def ecludDist(x, y):
    return np.sqrt(sum(np.square(np.array(x) - np.array(y))))


def manhattanDist(x, y):
    return np.sum(np.abs(x - y))


def cos(x, y):
    return np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y))


def clusterMean(dataset):
    return sum(np.array(dataset)) / len(dataset)


def randCenter(dataset, k):
    temp = []
    while len(temp) < k:
        index = np.random.randint(0, len(dataset)-1)
        if  index not in temp:
            temp.append(index)
    return np.array([dataset[i] for i in temp])


def orderCenter(dataset, k):
    return np.array([dataset[i] for i in range(k)])


def kMeans(dataset, dist, center, k):
    global flag
    #middle re
    all_kinds = []
    noisedataset=[]
    for _ in range(k):
        temp = []
        all_kinds.append(temp)
    #distance
    for i in dataset:
        temp = []
        for j in center:
            temp.append(dist(i, j))
        all_kinds[temp.index(min(temp))].append(i)

    for i in range(k):
        print('no'+str(i)+'group:', all_kinds[i], '\n')
    flag += 1
    print('************************'+str(flag)+'times***************************')
    for i in range(k):
        plt.scatter([j[0] for j in all_kinds[i]], [j[1] for j in all_kinds[i]], marker='*')
        # draw convex hull
        result = graham_scan(all_kinds[i])
        length = len(result)
        for j in range(0, length - 1):
            plt.plot([result[j][0], result[j + 1][0]], [result[j][1], result[j + 1][1]], c='r')
        plt.plot([result[0][0], result[length - 1][0]], [result[0][1], result[length - 1][1]], c='r')

        all_kinds[i] = addNoise(all_kinds[i], result)
        for j in range(0,len(all_kinds[i])):
            noisedataset.append(all_kinds[i][j])
    plt.grid()
    plt.show()


    center_ = np.array([clusterMean(i) for i in all_kinds])
    
    # if (center_ == center).all():
    #     print('end')
    # for i in range(k):
    #     all_kinds[i]=addNoise(all_kinds[i],results[i])
    #
    if flag==7 or (center_==center).all():
        print('end')
        for i in range(k):
            print('no' + str(i) + 'group:', center_[i], '\n')
            plt.scatter([j[0] for j in all_kinds[i]], [j[1] for j in all_kinds[i]], marker='*')

        plt.grid()
        plt.show()
    else:
        center = center_
        kMeans(noisedataset, dist, center, k)


def get_bottom_point(points):

    min_index = 0
    n = len(points)
    for i in range(0, n):
        if points[i][1] < points[min_index][1] or (
                points[i][1] == points[min_index][1] and points[i][0] < points[min_index][0]):
            min_index = i
    return min_index


def sort_polar_angle_cos(points, center_point):

    n = len(points)
    cos_value = []
    rank = []
    norm_list = []
    for i in range(0, n):
        point_ = points[i]
        point = [point_[0] - center_point[0], point_[1] - center_point[1]]
        rank.append(i)
        norm_value = math.sqrt(point[0] * point[0] + point[1] * point[1])
        norm_list.append(norm_value)
        if norm_value == 0:
            cos_value.append(1)
        else:
            cos_value.append(point[0] / norm_value)

    for i in range(0, n - 1):
        index = i + 1
        while index > 0:
            if cos_value[index] > cos_value[index - 1] or (
                    cos_value[index] == cos_value[index - 1] and norm_list[index] > norm_list[index - 1]):
                temp = cos_value[index]
                temp_rank = rank[index]
                temp_norm = norm_list[index]
                cos_value[index] = cos_value[index - 1]
                rank[index] = rank[index - 1]
                norm_list[index] = norm_list[index - 1]
                cos_value[index - 1] = temp
                rank[index - 1] = temp_rank
                norm_list[index - 1] = temp_norm
                index = index - 1
            else:
                break
    sorted_points = []
    for i in rank:
        sorted_points.append(points[i])

    return sorted_points


def vector_angle(vector):

    norm_ = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])
    if norm_ == 0:
        return 0

    angle = math.acos(vector[0] / norm_)
    if vector[1] >= 0:
        return angle
    else:
        return 2 * math.pi - angle


def coss_multi(v1, v2):

    return v1[0] * v2[1] - v1[1] * v2[0]


def graham_scan(points):
    bottom_index = get_bottom_point(points)
    bottom_point = points.pop(bottom_index)
    sorted_points = sort_polar_angle_cos(points, bottom_point)

    m = len(sorted_points)
    if m < 2:
        print("points not enough")
        return

    stack = []
    stack.append(bottom_point)
    stack.append(sorted_points[0])
    stack.append(sorted_points[1])

    for i in range(2, m):
        length = len(stack)
        top = stack[length - 1]
        next_top = stack[length - 2]
        v1 = [sorted_points[i][0] - next_top[0], sorted_points[i][1] - next_top[1]]
        v2 = [top[0] - next_top[0], top[1] - next_top[1]]

        while coss_multi(v1, v2) >= 0:
            stack.pop()
            length = len(stack)
            top = stack[length - 1]
            next_top = stack[length - 2]
            v1 = [sorted_points[i][0] - next_top[0], sorted_points[i][1] - next_top[1]]
            v2 = [top[0] - next_top[0], top[1] - next_top[1]]

        stack.append(sorted_points[i])

    return stack

def addNoise(cpoints,result):
    center=clusterMean(cpoints)
    didcon34=[]
    didcon24=[]
    didcon14=[]
    po44=[]
    po43=[]
    po32=[]
    po21=[]
    length=len(result)
    for i in range(0,length-1):
        difx=result[i][0]-center[0]
        dify=result[i][1]-center[1]
        didcon34.append([center[0]+difx*3/4,center[0]+difx*3/4])
        didcon24.append([center[0]+difx/2,center[0]+difx/2])
        didcon14.append([center[0]+difx/4,center[0]+difx/4])
    for point in cpoints:
        if(cover(point,didcon34)==0):
            po44.append(point)
        if(cover(point,didcon34)==1 and cover(point,didcon24)==0):
            po43.append(point)
        if (cover(point,didcon24)==1 and cover(point,didcon14)==0):
            po32.append(point)
        if (cover(point,didcon14)==1):
            po21.append(point)
    #add noise
    po44=laplace_mech(po44,1,1)
    po43=laplace_mech(po43,1,0.8)
    po32=laplace_mech(po32,1,0.5)
    po21=laplace_mech(po21,1,0.3)
    noisepoints=[]
    for i in range(0,len(po44)):
        noisepoints.append(po44[i])
    for i in range(0,len(po43)):
        noisepoints.append(po43[i])
    for i in range(0,len(po32)):
        noisepoints.append(po43[i])
    for i in range(0,len(po21)):
        noisepoints.append(po21[i])
    return noisepoints


def cross(point, rp1, rp2):
            x1 = rp1[0] - point[0]
            y1 = rp1[1] - point[1]
            x2 = rp2[0] - point[0]
            y2 = rp2[1] - point[1]
            return x1 * y2 - x2 * y1

def cover(point, results):
            length = len(results)
            results.append(results[0])
            results.append(results[1])
            results.append(results[2])
            for i in range(0, length + 1):
                if (cross(point, results[i], results[i + 1]) * cross(point, results[i + 1], results[i + 2]) < 0):
                    return 0
                else:
                    return 1


def noisyCount(sensitivety, epsilon):
    beta = sensitivety / epsilon
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta * np.log(1. - u2)
    else:
        n_value = beta * np.log(u2)
    #print(n_value)
    return n_value


def laplace_mech(data, sensitivety, epsilon):
    for i in range(len(data)):
        data[i] += noisyCount(sensitivety, epsilon)
    return data


#def read(file):


def main(k):
    x = [np.random.randint(0, 500) for _ in range(2000)]
    y = [np.random.randint(0, 500) for _ in range(2000)]
    points = [[i,j] for i, j in zip(x, y)]
    plt.plot(x, y, 'b.')
    plt.show()
    initial_center = randCenter(dataset=points, k=k)
    kMeans(dataset=points, dist=ecludDist, center=initial_center, k=k)

if __name__ == '__main__':
    main(7)