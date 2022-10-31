from domain.contracts.abstract_feature_extractor import AbstractFeatureExtractor

class KNN(AbstractFeatureExtractor):
    
    def get_most_frequent_element(l):
        return max(l, key=l.count) # counts how many times an element is present in a list : l [1, 1, 2] --> l.count [2, 2, 1] 1 is present 2 times, 1 is present 2 times, 2 is present 1 time

    def dist(x, y): # returns sqrt( sum (  (Ai - Bi)^2  ))
        return sum(
            [ 
                ( x_i - y_i ) ** 2 for x_i, y_i in zip(x, y)
            ] # zip : x = [1, 2, 3], y = ['a', 'b', 'c'], zip(x, y) = [ [1, 'a'], [2, 'b'], [3, 'c'] ]   #here 1 and a are PIXELS THAT WE WILL COMPARE
        )

    def get_training_distances_for_test_sample(X_train, test_sample):
        return [KNN.dist(train_sample, test_sample) for train_sample in X_train] # for every trained digit, we are going to find its distance from the test digit

    # 1 White/Black
    def knn(X_train, y_train, X_test, k=3): # for our point, it looks at ALL OTHER points, calculates distance with them, and takes the point with the smallest DISTANCE    | k : see which one returns best result (better as an odd parameter for no confusion if for example we have 2 classes and a point is confusing, but here we have a lot of classes)
    # 1) Comparing xtest sample with all xtrain elements
    # 2) saving top k candidates' indexes from xtrain 
    # 3) getting the candidtates from ytrain 
    # 4) adding them to ypred
        y_pred = []  # this is what we think every image is, so our output
        for test_sample_idx, test_sample in enumerate(X_test):
            # 1)
            training_distances = KNN.get_training_distances_for_test_sample(X_train, test_sample)
            # 2)
            sorted_distance_indices = [ # we want to sort INDICES acc. to the DISTANCE value, BUT we want to get the ELEMENT
                pair[0] # I only want the index: the first element of the tuple (index, distance)
                for pair in sorted(
                    enumerate(training_distances), # enumerate: l = ['a', 'b', 'c'], list(enumerate(l)) = [(0, 'a'), (1, 'b'), (2, 'c')] 
                    key= lambda x: x[1] # sort acc. to distance val
                )
            ] 
            #print(sorted_distance_indices) # shows : for each test sample, the distance between it and the 100 training samples
            # 3)
            candidates =[
                (y_train[idx]) 
                for idx in sorted_distance_indices[:k] # choose the top nearest k elements
            ]
            top_candidate = KNN.get_most_frequent_element(candidates)
            # 4)
            y_pred.append(top_candidate)
        return y_pred