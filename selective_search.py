import cv2
import numpy as np

def selective_obj_search(image, fastSearch = True):
    """selective search algotythm for choosing suspicious region"""
    # using multithreads
    cv2.setUseOptimized(True)
    cv2.setNumThreads(3)

    # selective search segmentation object
    SS = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    SS.setBaseImage(image)

    # high recall selective search is slow
    # low recall selective search is fast, but less effective
    if not fastSearch:
        SS.switchToSelectiveSearchQuality()
    else:
        SS.switchToSelectiveSearchFast()

    # run selective search segmentation on input image
    rectangles = SS.process()
    return rectangles


def subframes(frame, xcord, ycord, nededHight, nededWidth, steps = 3):
    """Choses subframes 50X50 from a single frame with step 3
    returns list of found subframes"""
    hight = frame.shape[0]
    width  = frame.shape[1] 
    if hight == nededHight and width == nededWidth:
        return [frame], [(xcord, ycord)]
    frames = np.array([frame[h_min : (h_min + nededHight), w_min : (w_min + nededWidth)]    \
             for h_min in range(0, hight - nededHight + 1, steps)       \
             for w_min in range(0, width - nededWidth + 1, steps)])
    coordinates = [(xcord + w_min, ycord + h_min)              \
             for h_min in range(0, hight - nededHight + 1, steps)       \
             for w_min in range(0, width - nededWidth + 1, steps)]
    return frames, coordinates


def objectSelector(image, nededWidth, nededHight, \
                    wholeFrame, numbRectanglesPersentage = 0.3, aditionalFrames = 9):
    """finds suspicious object that can be hands
    input: image - array;
    numbRectanglesPersentage - how many objects will the algorythm check;
    aditionalFrames - how bigger will the cheked picture be"""
    objectOfInterest = selective_obj_search(image)
    numbRect = int(numbRectanglesPersentage * len(objectOfInterest)) 
    image_set = []
    for i, rect in enumerate(objectOfInterest):
        if(i <= numbRect):
            x, y, w, h = rect
            coordinates = (x - aditionalFrames, y - aditionalFrames, w + 2*aditionalFrames, h + 2*aditionalFrames)
            if (x, y, w, h) == wholeFrame:
                continue
            center = (x + w//2, y + h//2)
            # reshaping image if it's too small
            if w < nededWidth:
                x = center[0] - nededWidth//2
                if x < 0:
                    x = 0
                w = nededWidth
                if x + w + aditionalFrames > wholeFrame[2]:
                    x = wholeFrame[2] - (w + aditionalFrames)
            if h < nededHight:
                y = center[1] - nededHight//2
                if y < 0:
                    y = 0
                h = nededHight
                if y + h + aditionalFrames > wholeFrame[3]:
                    y = wholeFrame[3] - (h + aditionalFrames)
            frames = image[y : (y + h + aditionalFrames), x : (x + w + aditionalFrames)]
            frames, cordsForFrames = subframes(frames, x, y, nededHight, nededWidth)
            image_set.append((frames, cordsForFrames))
        else:
            break
    return image_set


def plausibleHand(predictions, predictionError = 0.8):
    """finds objects, that are plausibly hands
    predictionError - plausibility error"""
    hands = []
    max_probability = 0
    max_position = 0
    probable_class = 2
    # maximum class pisibility
    for i, predict in enumerate(predictions):
        max_class = np.argmax(predict) 
        max_value = predict[max_class]
        if max_class == 0 or max_class == 1 or max_class == 3 or max_class == 4:
            hands.append((max_class, max_value, i))
        if max_value >= max_probability:
            max_probability = max_value
            max_position = i
            probable_class = max_class
    
    # check if class can be a hand
    max_hand_prob = 0
    probable_hand = 2
    hand_picture = 0
    for hand in hands:
        if max_hand_prob <= hand[1]:
            probable_hand = hand[0]
            max_hand_prob = hand[1]
            hand_picture = hand[2]

    # returns most plausible
    if max_hand_prob >= predictionError:
        return (probable_hand, max_hand_prob, hand_picture)
    else:
        return (probable_class, max_probability, max_position)


def plausiblePrediction(model, regtAndCordsSets, new_rows, new_colums):
    """returns maximum plausibility of a class, and its coordinates
    input: set of two sets: rectangles and their respective coordinates"""
    if len(regtAndCordsSets[0]) != len(regtAndCordsSets[0]):
        raise ValueError("Sizes of rectangle list and coordinates for rectangle list should be the same ")
    regtangleSet = regtAndCordsSets[0]
    regtangleSet = regtangleSet[:, :, : ,:1]
    
    # normalize and reshape data data
    regtangleSet = regtangleSet.astype("float32")
    regtangleSet /= 255
    rect = regtangleSet.reshape(regtangleSet.shape[0], new_rows, new_colums, 1)
    
    # predict class
    predictions = model.predict(rect)
    pr_class, prob, pos = plausibleHand(predictions)
    
    return (regtangleSet[pos], (regtAndCordsSets[1])[pos], pr_class, prob)