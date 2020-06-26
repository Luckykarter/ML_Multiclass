import cv2
import numpy as np

def recognize_from_video(model, labels):
    cap = cv2.VideoCapture(0)
    once = True
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        img_to_show = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.resize(img_to_show, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)

        frame = np.array(gray) * (1/255.0)
        # print(frame.shape)
        frame = np.expand_dims(frame, axis=0)
        # print(frame.shape)

        images = np.vstack([frame])
        classes = model.predict(images)
        if classes[0] > 0.5:
            obj = labels[1]
        else:
            obj = labels[0]

        # results = dict()
        # for i in range(len(labels)):
        #     results[labels[i]] = round(classes[0][i] * 100, 2)
        # # print(file, results)
        #
        # results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        # obj = ''
        # for item in results:
        #     obj += '{}: ({}%) '.format(item[0], item[1])

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (20, 40)
        fontScale = 0.5
        fontColor = (0, 0, 0)
        lineType = 2

        cv2.putText(img_to_show, obj,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        # Display the resulting frame
        cv2.imshow('Video', img_to_show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()