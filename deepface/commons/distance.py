import numpy as np

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))	

def findThreshold(model_name, distance_metric):
	
	base_threshold = {'cosine': 0.40, 'euclidean': 0.55, 'euclidean_l2': 0.75}
	
	thresholds = {
		'VGG-Face': {'cosine': 0.3828998554497957, 'euclidean': 0.5837138535827399, 'euclidean_l2': 0.8437118818678639},
		'OpenFace': {'cosine': 0.10, 'euclidean': 0.55, 'euclidean_l2': 0.55},
		'Facenet':  {'cosine': 0.6142496640305134, 'euclidean': 12.794783464670182, 'euclidean_l2': 1.0578221621432087},
		'DeepFace': {'cosine': 0.23, 'euclidean': 64, 'euclidean_l2': 0.64},
		'DeepID': 	{'cosine': 0.015, 'euclidean': 45, 'euclidean_l2': 0.17},
		'Dlib': 	{'cosine': 0.08838775356588767, 'euclidean': 0.5821402502891807, 'euclidean_l2': 0.4102203243676945},
		'ArcFace':  {'cosine': 0.620748013800026, 'euclidean': 5.081651272556998, 'euclidean_l2': 1.0885364669832316}
		}

	threshold = thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)
	
	return threshold