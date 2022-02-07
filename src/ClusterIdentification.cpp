#include "ClusterIdentification.hpp"

const std::vector<std::string> colors {"black", "red", "blue", "green", "cyan", 
    "yellow", "brown", "magenta"};

ClusterIdentification::ClusterIdentification(bool pltInput) {

    this->plt = new plotcpp::Plot(pltInput);
}

ClusterIdentification::~ClusterIdentification() {

    delete this->plt;
}

void ClusterIdentification::Setup(std::vector<cv::Vec2d> inputPoints) {

    
}

void ClusterIdentification::Setup(const std::string fileName) {

    std::ifstream inputFile(fileName);
    dlib::matrix<double> inputs;

    // read input from csv file input
    inputFile >> inputs;

    // necessary information about input data
    auto numSamples = inputs.nr();
    auto numFeatures = inputs.nc();
    int numClusters = 3;

    // pass input data to dlibs clustering algorithm
    std::string outputPlotName( "output_cluster_plot.png" );
    HierchicalClustering(inputs, numClusters, outputPlotName);

    inputFile.close();
}

void ClusterIdentification::PlotClusters(const Clusters &clusters, const std::string &name, const std::string &fileName) {

    plotcpp::Plot plt(true);
    plt.SetTerminal("png");
    plt.SetOutput(fileName);
    plt.SetTitle(name);
    plt.SetXLabel("x");
    plt.SetYLabel("y");
    plt.SetAutoscale();
    plt.GnuplotCommand("set grid");

    auto draw_state = plt.StartDraw2D<Coords::const_iterator>();
    for(auto& cluster : clusters) {

        std::stringstream params;
        params << "lc rgb '" << colors[cluster.first] << "' pt 7";
        plt.AddDrawing(draw_state, plotcpp::Points(cluster.second.first.begin(), cluster.second.first.end(),
            cluster.second.second.begin(), std::to_string(cluster.first) + " cls", params.str()));
    }

    plt.EndDraw2D(draw_state);
    plt.Flush();
}

template<typename T>
void ClusterIdentification::HierchicalClustering(const T &inputs, size_t numClusters, const std::string &name) {

    // distance calculations necessary for agglomeratice clustering algorithm
    dlib::matrix<double> distances(inputs.nr(), inputs.nr());
    for(long r = 0; r < distances.nr(); ++r) {

        for(long c = 0; c < distances.nc(); ++c) {

            distances(r, c) = length(subm(inputs, r, 0, 1, 2) - subm(inputs, c, 0, 1, 2));
        }
    }

    // call dlib's clustering algorithm and plot resulting clusters
    std::vector<unsigned long> clusters;
    bottom_up_cluster(distances, clusters, numClusters);
    Clusters plotClusters;
    for(long i = 0; i != inputs.nr(); ++i) {

        auto clusterIdx = clusters[i];
        plotClusters[clusterIdx].first.push_back(inputs(i, 0));
        plotClusters[clusterIdx].second.push_back(inputs(i, 1));
    }

    PlotClusters(plotClusters, "Agglomerative clustering", name);
}