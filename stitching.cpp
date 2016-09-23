#include "stitching.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/stitcher.hpp>
#include <opencv2/stitching/warpers.hpp>
#include <opencv2/stitching/detail/autocalib.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "CAMRefiner.h"

using namespace std;
using namespace cv;
using namespace cv::detail;

// Scales
double work_scale = 1;

// MultiBand Strength
float blend_strength = 10;

// Max Images
const int max_images = 10;

// GPU
bool try_use_gpu = false;

// Image Counter
static int imageCounter = 0;

// Globals
vector<Mat> global_images(max_images);
vector<CameraParams> cameras_global(max_images);
vector<CameraParams> refined_cameras_global(max_images);
vector<ImageFeatures> global_features(max_images);

// Detector
Ptr<GridAdaptedFeatureDetector> detector = new GridAdaptedFeatureDetector(new GFTTDetector(200,0.01,0,4),400);
//Ptr<GFTTDetector> detector = new GFTTDetector(500);

// Extractor
Ptr<SiftDescriptorExtractor> extractor = new SiftDescriptorExtractor();

void refineCameraParametersWithFeatures(vector<ImageFeatures> &featuresToMatch,vector<MatchesInfo> &pairwise_matches,vector<CameraParams> &camerasToRefine){
    printf("\n### Adjusting..!");
    Ptr<CAMRefiner> adjuster = new CAMRefiner();
    adjuster->setConfThresh(0.90f);
    adjuster->setMaxIterations(3);
    if (!(*adjuster)(featuresToMatch,pairwise_matches,camerasToRefine)) {
        // Process Error (never runs anyways)
    }
    
    adjuster.release();
}

cv::Mat startStitching(){
    // Release Extractor & Detector
    detector.release();
    extractor.release();
    
    printf("\n### Stitching..!");
    
    // Resizing
    global_images.resize(imageCounter);
    global_features.resize(imageCounter);
    
    vector<Mat> images = global_images;
    int num_images = static_cast<int>(images.size());
    
    refined_cameras_global.resize(num_images);
    vector<CameraParams> cameras = refined_cameras_global;
    refined_cameras_global.clear();
    
    // Start counting the seconds
    printf("\nNumber of Images: %i\n", num_images);
    
    int64 app_start_time = getTickCount();
    
    // Warper Creator
    Ptr<WarperCreator> warper_creator = new cv::CylindricalWarper();
    
    // Find median focal length
    double warped_image_scale;{
        
        vector<double> focals;
        
        for (size_t i = 0; i < cameras.size(); ++i) focals.push_back(cameras[i].focal);
        sort(focals.begin(), focals.end());
        
        if (focals.size() % 2 == 1) warped_image_scale = focals[focals.size() / 2];
        else                        warped_image_scale = (focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
        
    }
    
    // Rotation warper
    Ptr < RotationWarper > warper = warper_creator->create(warped_image_scale);
    warper_creator.release();
    
    vector < Mat > masks_warped(num_images);
    vector < Mat > images_warped(num_images);
    vector < cv::Point > corners(num_images);
    cv::Rect destROI;
    
    for (int img_idx = 0; img_idx < num_images; ++img_idx) {
        cv::Size img_size = global_images[img_idx].size();
        
        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);
        
        Mat xmap, ymap;
        // Create ROI for Corners
        cv::Rect roi = warper->buildMaps(img_size, K, cameras[img_idx].R, xmap, ymap);
        corners[img_idx] = roi.tl();
        
        // Remap Image
        remap(global_images[img_idx], images_warped[img_idx], xmap, ymap, INTER_LINEAR, BORDER_REFLECT);
        
        // Remap Mask
        remap(Mat(img_size, CV_8U, Scalar::all(255)), masks_warped[img_idx], xmap, ymap, INTER_NEAREST, BORDER_CONSTANT);
        
        // Build Destination ROI for Prepare Blender
        destROI |= warper->warpRoi(img_size, K, cameras[img_idx].R);
        
        global_images[img_idx].release();
        xmap.release();
        ymap.release();
        K.release();
    }
    warper.release();
    
    cout << "\n Warp Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency())   << " sec";
    
    float blend_width = sqrt( static_cast<float>( destROI.size().area() ) ) * blend_strength * 0.01f;
    int nBands = static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.);
    
    Ptr < Blender > blender = new MultiBandBlender(try_use_gpu, nBands);
    blender->prepare(destROI);
    
    cout << "\n Prepare Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency())   << " sec";
//    
//    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN);
//    compensator->feed(corners, images_warped, masks_warped);
//    
//    for (int i = 0; i < num_images; i++) {
//        compensator->apply(i, corners[i], images_warped[i], masks_warped[i]);
//    }
//    compensator.release();
    
    cout << "\n Comp Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency())   << " sec";
    
//    Ptr<SeamFinder> seamFinder = new DpSeamFinder(DpSeamFinder::COLOR);
//    seamFinder->find(images_warped, corners, masks_warped);
//    seamFinder.release();

    cout << "\n Seam Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency())   << " sec";
    
    for (int i = 0; i < num_images; i++) {
        blender->feed( images_warped[i], masks_warped[i], corners[i]);
    }
    
    // Reset all vectors
    images.clear();
    images_warped.clear();
    masks_warped.clear();
    corners.clear();
    cameras.clear();
    
    global_images.clear();
    cameras_global.clear();
    global_features.clear();
    refined_cameras_global.clear();
    
    // Resize vectors
    cameras_global.resize(max_images);
    refined_cameras_global.resize(max_images);
    global_images.resize(max_images);
    global_features.resize(max_images);
    
    // Reset image counter
    imageCounter = 0;
    
    cout << "\n Blend Starting, total time: " << ((getTickCount() - app_start_time) / getTickFrequency())   << " sec";
    // Blend images
    Mat result, result_mask;
    blender->blend(result, result_mask);
    result.convertTo(result, CV_8UC3);
    
    cout << "\nFinished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency())   << " sec";
    
    blender.release();
    return result;
}

void startMatchingFeatures(int index, Mat &image){
    // Start Index: 0
    printf("\n Running Image: #%d\n",index);
    
    if (!detector) {
        detector = new GridAdaptedFeatureDetector(new GFTTDetector(200,0.01,0,4),400);
    }
    
    if (!extractor) {
        extractor = new SiftDescriptorExtractor();
    }
    cv::Mat greyMat;
    cv::cvtColor(image, greyMat, cv::COLOR_BGR2GRAY);
    detector->detect(greyMat, global_features[index].keypoints); // Image Detect
    global_features[index].img_idx = index;
    global_features[index].img_size = greyMat.size();
    extractor->compute(greyMat, global_features[index].keypoints, global_features[index].descriptors); // Image Compute
    greyMat.release();
}

void startRefiningCameraParameters(){
    int64 refine_time = getTickCount();
    
    // Start refining camera parameters in background
    vector<CameraParams> cameras = cameras_global;
    cameras.resize(imageCounter);
    
    vector<ImageFeatures> features = global_features;
    features.resize(imageCounter);
    
    // Start matching & refining
    vector<MatchesInfo> pairwise_matches;
    BestOf2NearestMatcher matcher(try_use_gpu,0.50f);
    matcher(features,pairwise_matches);
    
    // Refine
    refineCameraParametersWithFeatures(features,pairwise_matches,cameras);
    
    // Storing the updated values in refined_cameras
    refined_cameras_global = cameras;
    
    // Clear & collect garbage
    matcher.collectGarbage();
    cameras.clear();
    features.clear();
    pairwise_matches.clear();
    
    cout << "\n Refine Finished, total time: " << ((getTickCount() - refine_time) / getTickFrequency())   << " sec";
}

void addCameraParameters(Mat &image, int HFOV, Mat rotationMatrix) {
    cameras_global[imageCounter].R = rotationMatrix;
    cameras_global[imageCounter].ppx = image.size().width / 2;
    cameras_global[imageCounter].ppy = image.size().height / 2;
    
    printf("\n ppx: %f\n",cameras_global[imageCounter].ppx);
    printf("\n ppy: %f\n",cameras_global[imageCounter].ppy);
    
    // [focal length in mm]*[resolution]/[sensor size in mm]
    cameras_global[imageCounter].focal = (4.12*image.size().height/4.54) * 1.10;
    printf("\n focal: %f",cameras_global[imageCounter].focal);
    
    image.copyTo(global_images[imageCounter]);
    
    imageCounter++;
    startMatchingFeatures(imageCounter-1,image);
    image.release();
}

int main(){
    // Load Images & Rotation Matrixes
    Mat rotationMatrix1 = (Mat_<float>(3,3) << 0.999983,
                                               0.004122,
                                               0.004101,
                           
                                              -0.003892,
                                              -0.049576,
                                               0.998763,
                           
                                               0.004320,
                                              -0.998762,
                                              -0.049559);
    
    Mat image;
    
    image = imread("IMG_001.JPG", IMREAD_COLOR ); // Read the file
    if(image.empty()){
        cout <<  "Could not open or find the image" << std::endl;
        return -1;
    }
    
    addCameraParameters(image, 0, rotationMatrix1);
    
    Mat rotationMatrix2 = (Mat_<float>(3,3) << 0.953141,
                                               0.016877,
                                              -0.302055,
                           
                                               0.302522,
                                              -0.058480,
                                               0.951347,
                           
                                              -0.001609,
                                              -0.998146,
                                              -0.060845);
    
    Mat image2 = imread("IMG_002.JPG", IMREAD_COLOR);
    addCameraParameters(image2, 0, rotationMatrix2);
    
    Mat rotationMatrix3 = (Mat_<float>(3,3) << 0.814439,
                                               0.022769,
                                              -0.579802,
                           
                                               0.579986,
                                              -0.062026,
                                               0.812262,
                           
                                              -0.017468,
                                              -0.997815,
                                              -0.063722);
    Mat image3 = imread("IMG_003.JPG", IMREAD_COLOR);
    addCameraParameters(image3, 0, rotationMatrix3);
    
    // Refine them
    startRefiningCameraParameters();
    
    // Warp & Blend
    Mat result = startStitching();
    
    return 1;
}
