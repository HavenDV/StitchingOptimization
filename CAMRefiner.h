#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include <opencv2/stitching/detail/matchers.hpp>

// Return struct form FindMaxSpanningTree function
using cv::detail::ImageFeatures;
using cv::detail::MatchesInfo;
using cv::detail::CameraParams;
using cv::detail::Graph;
using std::vector;
using cv::Mat;

struct SpanningTreeRV {
    Graph span_tree;
    vector<int> centers;
};

class CAMRefiner {
public:
    CAMRefiner(int num_params_per_cam = 4, int num_errs_per_measurement = 3);
    
    bool operator ()(const vector<ImageFeatures> &features, const vector<MatchesInfo> &pairwise_matches, vector<CameraParams> &cameras);
    

    void setConfThresh(double conf_thresh) {
        conf_thresh_ = conf_thresh;
    }
    void setTermCriteria(const cv::TermCriteria& term_criteria) {
        term_criteria_ = term_criteria;
    }
    
    double confThresh() const {
        return conf_thresh_;
    }
    cv::TermCriteria termCriteria() const {
        return term_criteria_;
    }
    
    void setMaxIterations(unsigned int value) {
        max_iterations = value;
    }
    unsigned int maxIterations() {
        return max_iterations;
    }
    
private:
    void setUpInitialCameraParams(const vector<CameraParams> &cameras);
    void obtainRefinedCameraParams(vector<CameraParams> &cameras) const;
    void calcError(Mat &err);
    void calcJacobian(Mat &jac);
    
    SpanningTreeRV findMaxSpanningTree(int num_images, const vector<MatchesInfo> &pairwise_matches);
    void calcDeriv(const Mat &err1, const Mat &err2, double h, Mat res);
    
    static std::vector< std::pair< int, int> > calculate_ranges(const std::vector<int> & s_tree, int num_items);
    
    Mat err1_, err2_;
    
    int num_images_;
    int total_num_matches_;
    
    int num_params_per_cam_;
    int num_errs_per_measurement_;
    
    const ImageFeatures *features_;
    const MatchesInfo *pairwise_matches_;
    
    // Threshold to filter out poorly matched image pairs
    double conf_thresh_;
    
    //Levenberg–Marquardt algorithm termination criteria
    cv::TermCriteria term_criteria_;
    
    // Camera parameters matrix (CV_64F)
    Mat cam_params_;
    
    // Connected images pairs
    vector<int> edges_;
    
    
    // Limit the number of iterations on the main loop;
    unsigned int max_iterations;
    
};
