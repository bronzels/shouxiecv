import cv2
from predict_on_spot_img import predict_on_img

def predict_on_video(video_path, spot_dict, model, class_indict, ret=True, save=True):
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_index = 0
    while ret:
        ret, image = cap.read()
        count += 1
        frame_index += 1

        if count == 5:
            count = 0
            
            new_image = predict_on_img(image, spot_dict, model, class_indict, save=False)

            #cv2.imshow('frame', new_image)
            #if cv2.waitKey() & 0xFF == ord('q'):
            #    break
            if save:
                filename = 'with_marking_predict_video_frame_{:04d}.jpg'.format(frame_index)
                cv2.imwrite(filename, new_image)
        
    #cv2.destroyAllWindows()
    cap.release()
