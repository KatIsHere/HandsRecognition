import cv2
from math import sqrt
from selective_search import objectSelector, plausiblePrediction

def overlap(dot1, dot2, width = 50, height = 50):
    """shows, how much two pictures overlap each other
    for distance uses euclidean metric"""
    max_distance = sqrt(width**2 + height**2)
    distance = sqrt((dot1[0] - dot2[0])**2 + (dot1[1] - dot2[1])**2)
    return distance / max_distance


def findHand(pred):
    """returns plausible hands (left and right) on a picture, and their coordinates in a 
    dictionary {coordinates : label}"""
    hands = {}
    for keyPredict in pred:
        if pred[keyPredict][0] == 1:
            hands[keyPredict] = ("Right Hand", pred[keyPredict][1])
        if pred[keyPredict][0] == 0:
            hands[keyPredict] = ("Left Hand", pred[keyPredict][1])
        if pred[keyPredict][0] == 3:
            hands[keyPredict] = ("Left Hand", pred[keyPredict][1])
        if pred[keyPredict][0] == 4:
            hands[keyPredict] = ("Right Hand", pred[keyPredict][1])
    return hands


def drawRectHands(image, hands, width = 50, height = 50):
    """ Draws rectangle over found hands on the image
    input: image - np.array representing an image
    hands - dictionary of hand objects"""
    for hand_coordinates in hands:
        x, y = hand_coordinates
        label = hands[hand_coordinates][0]
        cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, label, (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3 ,(100, 255, 100), 1,cv2.LINE_AA)


def overlapHandsDelete(hands, coeffSame = 0.8, coeffDiff = 0.5, errorAllowed = 0.00001):
    new_hands = hands.copy()
    for hand in hands:
        for hand_second in hands:
            #x = abs(hand[0] - hand_second[0]) + abs(hand[1] - hand_second[1])
            x = overlap((hand[0], hand[1]), (hand_second[0], hand_second[1]))
            if x < coeffSame and hands[hand][0] == hands[hand_second][0]   \
                    and hand != hand_second:
                if hand in new_hands and hand_second in new_hands:
                    del new_hands[hand_second] 
    
    for hand in hands:
        for hand_second in hands:
            if hand_second in new_hands:
                if x < coeffDiff and hands[hand][0] != hands[hand_second][0]:
                    error = (hands[hand][1] - hands[hand_second][1])
                    if  error > errorAllowed and hand in new_hands:
                        del new_hands[hand_second]
    return new_hands


def findObjects(model, img, width = 50, height = 50):
    """finds objects(hands) on an image"""
    regtangles = objectSelector(img, width, height, (0, 0, 189, 110))
    predictions = {}
    for rectangleCords in regtangles:
        picture, cords, classPosib, posib = plausiblePrediction(model, rectangleCords, height, width)
        predictions[cords] = (classPosib, posib)
    handsFound = findHand(predictions)
    handsFound = overlapHandsDelete(handsFound)
    return handsFound