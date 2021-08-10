"""
Panel detection class
"""

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
from matplotlib import cm
<<<<<<< HEAD
# from models.research.main_inference import load_image_into_numpy_array, run_inference_for_single_image, \
#     vis_util, delete_over_lapping, label_map_util
=======
from pv_learning.main_inference import load_image_into_numpy_array, run_inference_for_single_image, \
    vis_util, delete_over_lapping_in_image, label_map_util, filter_score
>>>>>>> 9dcbc9a593444238240087d29d3b26275f4a4ad7
import requests
from PIL import Image
from os import path

<<<<<<< HEAD
# panel_seg_model_path = path.join(path.dirname(__file__), 'VGG16Net_ConvTranpose_complete.h5')
# panel_classification_model_path = path.join(path.dirname(__file__), 'VGG16_classification_model.h5')
# output_directory = 'inference_graph'
# train_record_path = "../models/research/object_detection/pv-learning/train.record"
# test_record_path = "../models/research/object_detection/pv-learning/test.record"
# labelmap_path = "../models/research/object_detection/pv-learning/labelmap.pbtxt"
# base_config_path = "../models/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config"
=======
panel_seg_model_path = path.join(path.dirname(__file__), 'VGG16Net_ConvTranpose_complete.h5')
panel_classification_model_path = path.join(path.dirname(__file__), 'VGG16_classification_model.h5')
base_dir = path.abspath(path.join(path.dirname(__file__), ".."))
#
# output_directory = 'inference_graph'
# train_record_path = "pv_learning/train.record"
# test_record_path = "pv_learning/test.record"
# labelmap_path = "pv_learning/labelmap.pbtxt"
# base_config_path = "pv_learning/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.config"
>>>>>>> 9dcbc9a593444238240087d29d3b26275f4a4ad7
# dir_path = path.abspath(path.join(__file__, "..", ".."))
# tf.gfile = tf.io.gfile
# category_index = \
#     label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
<<<<<<< HEAD

# tf.keras.backend.clear_session()
# model = tf.saved_model.load(f'../models/research/object_detection/{output_directory}/saved_model')

=======
#
# tf.keras.backend.clear_session()
# model = tf.saved_model.load(f'pv_learning/{output_directory}/saved_model')

train_record_path = "pv_learning/train.record"
test_record_path = "pv_learning/test.record"
labelmap_path = "pv_learning/labelmap.pbtxt"
base_config_path = "pv_learning/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.config"
>>>>>>> 9dcbc9a593444238240087d29d3b26275f4a4ad7

class PanelDetection:
    '''
    A class for training a deep learning architecture, 
    detecting solar arrays from a satellite image, performing spectral
    clustering, and predicting the Azimuth.
    '''
    def __init__(self, model_file_path='./VGG16Net_ConvTranpose_complete.h5',
<<<<<<< HEAD
                 classifier_file_path='./VGG16_classification_model.h5'):
        
=======
                 classifier_file_path='./VGG16_classification_model.h5',):
>>>>>>> 9dcbc9a593444238240087d29d3b26275f4a4ad7
        # This is the model used for detecting if there is a panel or not
        self.classifier = load_model(classifier_file_path,
                                     custom_objects=None,
                                     compile=False)

        self.model = load_model(model_file_path,
                                custom_objects=None,
                                compile=False)
<<<<<<< HEAD
        self.classifier = None
        self.model = None
=======
        output_directory = f'{base_dir}/pv_learning/inference_graph'
        labelmap_path = f"{base_dir}/pv_learning/labelmap.pbtxt"
        self.dir_path = path.abspath(path.join(__file__, "..", ".."))
        tf.gfile = tf.io.gfile
        tf.keras.backend.clear_session()
        self.solar_array_detection_model = tf.saved_model.load(f'{output_directory}/saved_model')
        self.category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)


>>>>>>> 9dcbc9a593444238240087d29d3b26275f4a4ad7

    def generateSatelliteImage(self, latitude, longitude,
                               file_name_save, google_maps_api_key):
        """
        Generates satellite image via Google Maps, using the passed lat-long coordinates.
        
        Parameters
        -----------
        latitude: Float. Latitude coordinate of the site.
        longitude: Float. Longitude coordinate of the site.
        file_name_save: String. File path that we want to save the image to. PNG file.
        google_maps_api_key: String. Google Maps API Key for automatically 
            pulling satellite images.
                
        Returns
        -----------    
        Returned satellite image.
        """
        #Check input variable for types
        if type(latitude) != float:
            raise TypeError("latitude variable must be of type float.")
        if type(longitude) != float:
            raise TypeError("longitude variable must be of type float.")    
        if type(file_name_save) != str:
            raise TypeError("file_name_save variable must be of type string.")
        if type(google_maps_api_key) != str:
            raise TypeError("google_maps_api_key variable must be of type string.")
        #Build up the lat_long string from the latitude-longitude coordinates
        lat_long = str(latitude)+ ", "+ str(longitude)
        # get method of requests module 
        # return response object 
        r = requests.get("https://maps.googleapis.com/maps/api/staticmap?maptype=satellite&center=" + lat_long + "&zoom=18&size=35000x35000&key="+google_maps_api_key,
                         verify= False)    
        #Raise an exception if the satellite image is not successfully returned
        if r.status_code != 200:
            raise ValueError("Response status code " + str(r.status_code) + ": Image not pulled successfully from API.")
        # wb mode is stand for write binary mode 
        f = open(file_name_save, 'wb')     
        # r.content gives content, 
        # in this case gives image 
        f.write(r.content)
        # close method of file object 
        # save and close the file 
        f.close()
        #Read in the image and return it via the console
        return Image.open(file_name_save)

    def get_classification_and_score_from_long_and_lat(self, latitude, longitude, google_maps_api_key,
                                                       file_name="image.jpeg", inference_save_dir=None,
                                                       regular_file_dir=None):
        """
        Generates inferred satellite image via Google Maps, using the passed lat-long coordinates.

        Parameters
        -----------
        latitude: Float. Latitude coordinate of the site.
        longitude: Float. Longitude coordinate of the site.
        google_maps_api_key: String. Google Maps API Key for automatically
            pulling satellite images.
        file_name: String. File path that we want to save the image to. PNG file.
        inference_save_dir: String. Folder path for inferred images to save in.
        regular_file_dir: String. Folder path for images pre-inference to be saved in.

        Returns
        -----------
        Returns None.
        """
        if regular_file_dir is None:
<<<<<<< HEAD
            regular_file_dir = "{0}/{1}".format(path.abspath(path.join(dir_path, "clean")), file_name)
        if inference_save_dir is None:
            inference_save_dir = "{0}/{1}".format(path.abspath(path.join(dir_path, "inferred")), file_name)
        # self.generateSatelliteImage(latitude=latitude, longitude=longitude,
        #                             file_name_save=regular_file_dir,
        #                             google_maps_api_key=google_maps_api_key)

        image = load_image_into_numpy_array(regular_file_dir)

        output_dict = run_inference_for_single_image(model, image)
        output_dict = delete_over_lapping(output_dict)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            agnostic_mode=False,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        img = Image.fromarray(image)
        img.save("{}".format(inference_save_dir))
=======
            regular_file_dir = "{0}/{1}".format(path.abspath(path.join(self.out_dir_path, "clean")), file_name)
        if inference_save_dir is None:
            inference_save_dir = "{0}/{1}".format(path.abspath(path.join(self.out_dir_path, "inferred")), file_name)
        self.generateSatelliteImage(latitude=latitude, longitude=longitude,
                                    file_name_save=regular_file_dir,
                                    google_maps_api_key=google_maps_api_key)

        image = load_image_into_numpy_array(regular_file_dir)

        output_dict = run_inference_for_single_image(self.solar_array_detection_model, image)
        output_dict = filter_score(output_dict, 0.9)
        output_dict = delete_over_lapping_in_image(output_dict)
        if output_dict['num_detections'] <= 0:
            pass
        else:
            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                self.category_index,
                agnostic_mode=False,
                instance_masks=None,
                use_normalized_coordinates=True,
                line_thickness=8)
            img = Image.fromarray(image)
            img.save("{}".format(inference_save_dir))
>>>>>>> 9dcbc9a593444238240087d29d3b26275f4a4ad7
        print("{} inference made".format(path.basename(inference_save_dir)))


    def diceCoeff(self, y_true, y_pred, smooth=1):
        """
        This function is used as the metric of similarity between the 
        predicted mask and ground truth. 
        
        Parameters
        -----------
        y_true - (numpy array of floats) 
            the true mask of the image                        
        y_pred - (numpy array  of floats) 
            the predicted mask of the data
        smooth - (int): 
            a parameter to ensure we are not dividing by zero and also a smoothing parameter. 
            For back propagation. If the prediction is hard threshold to 0 and 1, it is difficult to back
            propagate the dice loss gradient. We add this parameter to actually smooth out the loss function, 
            making it differentiable.
        
        Returns
        -----------
        dice: - float: retuns the metric of similarity between prediction and ground truth
        """
        #Ensure that the inputs are of the correct type
        if type(y_true) != np.ndarray:
            raise TypeError("Variable y_true should be of type np.ndarray.")
        if type(y_pred) != np.ndarray:
            raise TypeError("Variable y_pred should be of type np.ndarray.")
        if type(smooth) != int:
            raise TypeError("Variable smooth should be of type int.")
        #If variable types are correct, continue with function
        intersection = K.sum(y_true * y_pred, axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
        return dice

    
    def diceCoeffLoss(self, y_true, y_pred):
        """
        This function is a loss function that can be used when training the segmentation model.
        This loss function can be used in place of binary crossentropy,
        which is the current loss function in the training stage     
        
        Parameters
        -----------
        y_true - (numpy array of floats) 
            the true mask of the image                        
        y_pred - (numpy array of floats)
            the predicted mask of the data
        
        Returns
        -----------
        float: retuns the loss metric between prediction and ground truth
        
        """
        #Ensure that the inputs are of the correct type
        if type(y_true) != np.ndarray:
            raise TypeError("Variable y_true should be of type np.ndarray.")
        if type(y_pred) != np.ndarray:
            raise TypeError("Variable y_pred should be of type np.ndarray.")
        return 1-self.dice_coef(y_true, y_pred)
    

    def testBatch(self, test_data, test_mask=None, BATCH_SIZE = 16, model =None):
        """
        This function is used to predict the mask of a batch of test satellite images.
        Use this to test a batch of images greater than 4
        
        Parameters
        -----------
        'test_data': (nparray float) 
            the satellite images                        
        'test_mask': (nparray int/float)  
            the mask ground truth corresponding to the test_data
        'batch_size': (int)  
            the batch size of the test_data. 
        'model': (tf.keras.model.object)
            a custom model can be provided as input or we can use the initialized model
        
        Returns
        -----------
        'test_res': (nparray float) 
            retuns the predicted masks.
        'accuracy': (float) 
            returns the accuarcy of prediction as compared with the ground truth if provided
        """
        #Ensure that the inputs are of the correct type
        if type(test_data) != np.ndarray:
            raise TypeError("Variable test_data should be of type np.ndarray.")
        if type(BATCH_SIZE) != int:
            raise TypeError("Variable BATCH_SIZE should be of type int.")        
        test_datagen = image.ImageDataGenerator(rescale=1./255,dtype='float32')
        test_image_generator = test_datagen.flow(
                test_data,
                batch_size = BATCH_SIZE, shuffle=False)
        if model != None:
            test_res = model.predict(test_image_generator)
        else :
            test_res = self.model.predict(test_image_generator)
        if test_mask != None: 
            test_mask = test_mask/np.max(test_mask)
            accuracy = self.dice_coef(test_mask,test_res)  
            return test_res,accuracy
        else:
            return test_res

    def testSingle(self, test_data, test_mask=None,  model =None):
        """
        This function is used to predict the mask corresponding to a single test image. 
        It takes as input the test_data (a required parameter) and two non-required parameters- test_mask and model
        
        Use this to test a single image.

        Parameters
        -----------
        'test_data': (nparray int or float)  
            the satellite image. dimension is (640,640,3) or (a,640,640,3)                     
        'test_mask': (nparray int/flaot)  
            the ground truth of what the mask should be 
        'model': (tf.keras model object) 
            a custom model can be provided as input or we can use the initialized model
        
        Returns
        -----------
        'test_res': (nparray float)  
            retuns the predicted mask of the single image. The dimension is (640,640 or (a,640,640))
        'accuracy': (float) 
            returns the accuarcy of prediction as compared with the ground truth if provided
           
        """
        #check that the inputs are correct
        if type(test_data) != np.ndarray:
            raise TypeError("Variable test_data must be of type Numpy ndarray.")
        #Test that the input array has 2 to 3 channels
        if (len(test_data.shape) > 3) | (len(test_data.shape) < 2):
            raise ValueError("Numpy array test_data shape should be 2 or 3 dimensions.")
        #Once the array passes checks, run the sequence
        test_data = test_data/255
        test_data = test_data[np.newaxis, :]
        if model != None:
            test_res = model.predict(test_data)
        else :
            test_res = self.model.predict(test_data)
            test_res = (test_res[0].reshape(640,640))
        if test_mask != None: 
            test_mask = test_mask/np.max(test_mask)
            accuracy = self.dice_coef(test_mask,test_res)  
            return test_res,accuracy
        else:
            return test_res        
        

    def hasPanels(self, test_data):
        """
        This function is used to predict if there is a panel in an image or not. 
        Note that it uses a saved classifier model we have trained and not the 
        segmentation model.       
        
        Parameters
        -----------
        test_data: (nparray float or int) 
            the satellite image. The shape should be [a,640,640,3] where 
            'a' is the number of data or (640,640,3) if it is a single image
                                       
        Returns
        -----------
        Boolean. Returns True if solar array is detected in an image, and False otherwise.
        """
        #Check that the input is correct
        if type(test_data) != np.ndarray:
            raise TypeError("Variable test_data must be of type Numpy ndarray.")
        #Test that the input array has 3 to 4 channels
        if (len(test_data.shape) > 4) | (len(test_data.shape) < 3):
            raise ValueError("Numpy array test_data shape should be 3 dimensions if a single image, or 4 dimensions if a batch of images.")        
        test_data = test_data/255
        #This ensures the first dimension is the number of test data to be predicted
        if test_data.ndim == 3:
            test_data = test_data[np.newaxis, :]
        prediction = self.classifier.predict(test_data)
        #index 0 is for no panels while index 1 is for panels
        if prediction[0][1] > prediction[0][0]:
            return True 
        else:
            return False
        

    def detectAzimuth(self, in_img, number_lines=5):
        """
        This function uses canny edge detection to first extract the edges of the input image. 
        To use this function, you have to first predict the mask of the test image 
        using testSingle function. Then use the cropPanels function to extract the solar 
        panels from the input image using the predicted mask. Hence the input image to this 
        function is the cropped image of solar panels.
        
        After edge detection, Hough transform is used to detect the most dominant lines in
        the input image and subsequently use that to predict the azimuth of a single image
  
        Parameters
        -----------
        in_img: (nparray uint8) 
            The image containing the extracted solar panels with other pixels zeroed off. Dimension is [1,640,640,3]
        number_lines: (int)  
            This variable tells the function the number of dominant lines it should examine.
            We currently inspect the top 10 lines.
            
        Returns
        -----------
        azimuth: (int) 
            The azimuth of the panel in the image.
        """
        #Check that the input variables are of the correct type
        if type(in_img) != np.ndarray:
            raise TypeError("Variable in_img must be of type Numpy ndarray.")
        if type(number_lines) != int:
            raise TypeError("Variable number_lines must be of type int.")
        #Run through the function
        edges = cv2.Canny(in_img[0],50,150,apertureSize=3)
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
        h, theta, d = hough_line(edges, theta=tested_angles)
        origin = np.array((0, edges.shape[1]))
        ind =0
        azimuth = 0
        az = np.zeros((number_lines))
        # Classic straight-line Hough transform
        # Set a precision of 0.5 degree.        
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=number_lines, threshold =0.25*np.max(h))):
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
                
            deg_ang = int(np.rad2deg(angle))
            if deg_ang >= 0:
                az[ind] = 90+deg_ang
            else:
                az[ind] = 270 + deg_ang
            ind =ind+1
        unique_elements, counts_elements = np.unique(az, return_counts=True)
        check = counts_elements[np.argmax(counts_elements)]
        if check == 1:
            for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=1, threshold =0.25*np.max(h))):
                deg_ang = int(np.rad2deg(angle))
                if deg_ang >= 0:
                    azimuth = 90+deg_ang
                else:
                    azimuth = 270 + deg_ang
        else:
            azimuth = (unique_elements[np.argmax(counts_elements)])
        return azimuth    

    
    def cropPanels(self, test_data, test_res):
        """
        This function basically isolates regions with solar panels in a 
        satellite image using the predicted mask. It zeros out other pixels that does not 
        contain a panel.
        You can use this for a single test data or multiple test data. 
        
        Parameters 
        ----------
        test_data:  (nparray float)
            This is the input test data. This can be a single image or multiple image. Hence the 
            dimension can be (640,640,3) or (a,640,640,3)
        test_res:   (nparray float) 
            This is the predicted mask of the test images passed as an input and used to crop out the 
            solar panels. dimension is (640,640)
        
        Returns 
        ----------
        new_test_res: (nparray uint8) 
            This returns images here the solar panels have been cropped out and the background zeroed. 
            It has the same shape as test data.  The dimension is [a,640,640,3] where a is the number of
            input images
            
        """
        #Check that the input variables are of the correct type
        if type(test_data) != np.ndarray:
            raise TypeError("Variable test_data must be of type Numpy ndarray.")
        if type(test_res) != np.ndarray:
            raise TypeError("Variable test_res must be of type Numpy ndarray.")            
        #Convert the test_data array from 3D to 4D
        if test_data.ndim == 3:
            test_data = test_data[np.newaxis, :]
        new_test_res = np.uint8(np.zeros((test_data.shape[0],640,640,3)))
        for ju in np.arange(test_data.shape[0]):
            try:
                in_img = test_res[ju].reshape(640,640)
            except:
                in_img = test_res.reshape(640,640)
            in_img[in_img < 0.9] = 0
            in_img[in_img >= 0.9] = 1
            in_img = np.uint8(in_img)
            test2 = np.copy(test_data[ju])
            test2[(1-in_img).astype(bool),0] = 0
            test2[(1-in_img).astype(bool),1] = 0
            test2[(1-in_img).astype(bool),2] = 0
            new_test_res[ju] = test2    
        return new_test_res
        
    
    def plotEdgeAz(self, test_results, no_lines=5, 
                    no_figs=1, save_img_file_path = None,
                    plot_show = False):
        """
        This function is used to generate plots of the image with its azimuth
        It can generate three figures or one. For three figures, that include the 
        input image, the hough transform space and the input images with detected lines.
        For single image, it only outputs the input image with detected lines.
        
        Parameters 
        ----------
        test_results: (nparray float64 or unit8) 
            8-bit input image. This variable represents the predicted images from the segmentation model. Hence the 
            dimension must be [a,b,c,d] where [a] is the number of images, [b,c] are the dimensions
            of the image - 640 x 640 in this case and [d] is 3 - RGB
        no_lines: (int) 
            default is 10. This variable tells the function the number of dominant lines it should examine.                  
        no_figs: (int) 
            1 or 3. If the number of figs is 1. It outputs the mask with Hough lines and the predicted azimuth
            However, if the number of lines is 3, it gives three plots. 
                1. The input image,
                2. Hough transform search space
                3. Unput image with houghlines and the predicted azimuth
                          
        save_img_file_path: (string) 
            You can pass as input the location to save the plots
        plot_show: Boolen: If False, it will supress the plot as an output and just save the  plots in a folder
        
        Returns 
        ----------
        Plot of the masked image, with detected Hough Lines and azimuth estimate.
        """
        #Check that the input variables are of the correct type
        if type(test_results) != np.ndarray:
            raise TypeError("Variable test_results must be of type Numpy ndarray.")
        if type(no_lines) != int:
            raise TypeError("Variable no_lines must be of type int.")  
        if type(no_figs) != int:
            raise TypeError("Variable no_figs must be of type int.")              
        if type(plot_show) != bool:
            raise TypeError("Variable no_figs must be of type boolean.")  
        
        for ii in np.arange(test_results.shape[0]):
            #This changes the float64 to uint8
            if (test_results.dtype is np.dtype(np.float64)):
                in_img = test_results[ii].reshape(640,640)
                in_img[in_img < 0.9] = 0
                in_img[in_img >= 0.9] = 1
                in_img = np.uint8(in_img)

            in_img = test_results[ii]
            #Edge detection
            edges = cv2.Canny(in_img,50,150,apertureSize=3)
            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
            h, theta, d = hough_line(edges, theta=tested_angles)
            az = np.zeros((no_lines))
            origin = np.array((0, edges.shape[1]))
            ind =0
            # Generating figure 1            
            fig, ax = plt.subplots(1, no_figs, figsize=(10, 6))
            if no_figs == 1:
                ax.imshow(edges)# cmap=cm.gray)
                for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=no_lines, threshold =0.25*np.max(h))):
                    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
                    deg_ang = int(np.rad2deg(angle))
                    if deg_ang >= 0:
                        az[ind] = 90+deg_ang
                    else:
                        az[ind] = 270 + deg_ang
                    ind =ind+1
                    ax.plot(origin, (y0, y1), '-r')
                ax.set_xlim(origin)
                ax.set_ylim((edges.shape[0], 0))
                ax.set_axis_off()
                unique_elements, counts_elements = np.unique(az, return_counts=True)
            
                check = counts_elements[np.argmax(counts_elements)]
                
                if check == 1:
                    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=1, threshold =0.25*np.max(h))):
                        deg_ang = int(np.rad2deg(angle))
                        if deg_ang >= 0:
                            azimuth = 90+deg_ang
                        else:
                            azimuth = 270 + deg_ang
                else:
                    azimuth = (unique_elements[np.argmax(counts_elements)])
                    #print(np.asarray((unique_elements, counts_elements)))
                    ax.set_title('Azimuth = %i' %azimuth)
                #save the image
                if save_img_file_path != None:
                    plt.savefig(save_img_file_path + '/crop_mask_az_'+str(ii),
                                dpi=300)
                #Show the plot if plot_show = True
                if plot_show == True:
                    plt.tight_layout()
                    plt.show()     
            elif no_figs == 3:
                ax = ax.ravel()

                ax[0].imshow(in_img, cmap=cm.gray)
                ax[0].set_title('Input image')
                ax[0].set_axis_off()
    

                ax[1].imshow(np.log(1 + h),
                    extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                    cmap=cm.gray, aspect=1/1.5)
                ax[1].set_title('Hough transform')
                ax[1].set_xlabel('Angles (degrees)')
                ax[1].set_ylabel('Distance (pixels)')
                ax[1].axis('image')

                ax[2].imshow(in_img)# cmap=cm.gray)
                origin = np.array((0, edges.shape[1]))
                ind =0
                for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=no_lines, threshold =0.25*np.max(h))):
                    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
                
                    deg_ang = int(np.rad2deg(angle))
                    if deg_ang >= 0:
                        az[ind] = 90+deg_ang
                    else:
                        az[ind] = 270 + deg_ang
                    ind =ind+1
                    ax.plot(origin, (y0, y1), '-r')
                ax[2].set_xlim(origin)
                ax[2].set_ylim((edges.shape[0], 0))
                ax[2].set_axis_off()
                unique_elements, counts_elements = np.unique(az, return_counts=True)
            
                check = counts_elements[np.argmax(counts_elements)]
                
                if check == 1:
                    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=1, threshold =0.25*np.max(h))):
                        deg_ang = int(np.rad2deg(angle))
                        if deg_ang >= 0:
                            azimuth = 90+deg_ang
                        else:
                            azimuth = 270 + deg_ang
                else:
                    azimuth = (unique_elements[np.argmax(counts_elements)])
                    #print(np.asarray((unique_elements, counts_elements)))
                    ax[2].set_title('Azimuth = %i' %azimuth)
                #save the image
                if save_img_file_path != None:
                    plt.savefig(save_img_file_path + '/crop_mask_az_'+str(ii),
                                dpi=300)
                #Show the plot if plot_show = True
                if plot_show == True:
                    plt.tight_layout()
                    plt.show() 
            else:
                print("Enter valid parameters")
                

    def clusterPanels(self, test_mask, fig=False):
        '''
        This function uses connected component algorithm to cluster the panels

        Parameters
        ----------
        test_mask : (bool) or (float)
            The predicted mask. Dimension is (640,640) or can be converted to RGB (640,640,3)
        fig : (bool)
            shows the clustering image if fig = True

        Returns
        -------
        (uint8)
        Masked image containing detected clusters each of dimension(640,640,3)
        
        (uint8)
        The optimal number of clusters
        '''
        #Check that the input variables are of the correct type
        if type(test_mask) != np.ndarray:
            raise TypeError("Variable test_mask must be of type Numpy ndarray.")
        if type(fig) != bool:
            raise TypeError("Variable fig must be of type bool.")        
        #Continue running through the function if all the inputs are correct
        if (len(test_mask.shape) < 3):
            test_mask = cv2.cvtColor(test_mask,cv2.COLOR_GRAY2RGB)
        test_mask =  test_mask.reshape(640,640,3) 
        # Converting those pixels with values 0-0.5 to 0 and others to 1
        img = cv2.threshold(test_mask, 0.5, 1, cv2.THRESH_BINARY)[1]
        # Applying cv2.connectedComponents() 
        num_labels, labels = cv2.connectedComponents(img[:,:,2].reshape(640,640))        
        # Map component labels to hue val, 0-179 is the hue range in OpenCV
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        # Converting cvt to BGR
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        # set background label to black
        labeled_img[label_hue==0] = 0
        #Initialize each clusters
        clusters = np.uint8(np.zeros((num_labels-1, 640, 640,3)))
        #starting from 1 to ignore background
        for i in np.arange(1,num_labels):
            clus = np.copy(test_mask)
            c_mask = labels==i
            #clus_label = np.zeros((640,640,3))
            clus[(1-c_mask).astype(bool),0] = 0
            clus[(1-c_mask).astype(bool),1] = 0
            clus[(1-c_mask).astype(bool),2] = 0
            #clus_label = np.stack((clus_label,)*3, axis=-1)
            clusters[i-1] = clus
        # Loop through each cluster, and detect number of non-zero values
        # in each cluster.
        clusters_list_keep = []
        for cluster_number in range(clusters.shape[0]):
            cluster = clusters[cluster_number]
            # Get the number of non-zero values as a ratio of total pixels
            pixel_count = len(cluster[cluster>0])
            total_pixels = cluster.shape[0] * cluster.shape[1] * cluster.shape[2]
            # Must greater than 3% non-zero pixels or we omit the cluster
            print(pixel_count / total_pixels)
            if (pixel_count / total_pixels) >= 0.0015:
                clusters_list_keep.append(cluster_number)
        # Filter clusters
        clusters = clusters[clusters_list_keep]
        if fig == True:
                #Showing Image after Component Labeling
                plt.figure()
                plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title("Image after Component Labeling")
                plt.show()
        return len(clusters),clusters
<<<<<<< HEAD
=======

p = PanelDetection()
>>>>>>> 9dcbc9a593444238240087d29d3b26275f4a4ad7
