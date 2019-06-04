import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.features2d.*;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class ImageSimilarity {
    private int NumOfKeyPairs = 25;
    private float properKeyPairCheck = 0.1f;

//openCV SIFT to extract KeyPoints
    public MatOfKeyPoint ExtractKeyPoints(Mat srcImage) {
        MatOfKeyPoint srcKeyPoints = new MatOfKeyPoint();
        FeatureDetector siftFeatureDetector = FeatureDetector.create(FeatureDetector.SIFT);
        siftFeatureDetector.detect(srcImage, srcKeyPoints);
        return srcKeyPoints;
    }
//poenCV SIFT to extract descriptors
    public MatOfKeyPoint ExtractDescriptors(Mat srcImage, MatOfKeyPoint srcKeyPoints) {
        MatOfKeyPoint srcKeyPointsDescriptors = new MatOfKeyPoint();
        DescriptorExtractor siftDescriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SIFT);
        siftDescriptorExtractor.compute(srcImage, srcKeyPoints, srcKeyPointsDescriptors);
        return srcKeyPointsDescriptors;
    }

    public List<DMatch> KeyPointPairsCreation(MatOfKeyPoint firstImageDescriptors, MatOfKeyPoint secondImageDescriptors) {
        MatOfDMatch matchesOfFrst = new MatOfDMatch();
        MatOfDMatch matchesOfScnd = new MatOfDMatch();
//openCV matcher
        DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
        descriptorMatcher.match(firstImageDescriptors, secondImageDescriptors, matchesOfFrst);
        descriptorMatcher.match(secondImageDescriptors, firstImageDescriptors, matchesOfScnd);
        return matchesOfFrst.toList().stream()
                .filter(match -> matchesOfScnd.toList().stream().  // mutual pointers of points
                        anyMatch(otherMatch -> match.queryIdx == otherMatch.trainIdx && match.trainIdx == otherMatch.queryIdx))
                .collect(Collectors.toList());
// take only mutual matches
    }

    public List<DMatch> FilterMatchedKeyPoints(List<DMatch> matches, KeyPoint[] firstImgKeyPoints, KeyPoint[] secondImgKeyPoints) {
        LinkedList<DMatch> goodMatchesList = new LinkedList<>();
        Map<Integer, List<Integer>> frstImgPointIdxWithNeighorhood2D = ExtractNeighbourhoods(matches.stream()
                .map(match -> match.queryIdx).collect(Collectors.toList()), firstImgKeyPoints);
        Map<Integer, List<Integer>> scndImgPointIdxWithNeighorhood2D = ExtractNeighbourhoods(matches.stream()
                        .map(match -> match.trainIdx).collect(Collectors.toList()), secondImgKeyPoints);
        for(DMatch match : matches) {
            int numIntegralMatches = 0;

            for(DMatch otherMatch : matches) {
                //mutual  pairs  in neighbors
                boolean otherMatchLeftIsNeighbourhood = frstImgPointIdxWithNeighorhood2D.get(match.queryIdx).contains(otherMatch.queryIdx);
                boolean otherMatchRightIsNeighbourhood = scndImgPointIdxWithNeighorhood2D.get(match.trainIdx).contains(otherMatch.trainIdx);

                if(otherMatchLeftIsNeighbourhood && otherMatchRightIsNeighbourhood) {
                    numIntegralMatches++;
                }
            }
            if((float) numIntegralMatches / (float) NumOfKeyPairs >= properKeyPairCheck) {
                goodMatchesList.addLast(match);
            }
        }
        return goodMatchesList;
    }

    private Map<Integer, List<Integer>> ExtractNeighbourhoods(List<Integer> pointIdxs, KeyPoint[] imgKeyPoints) {
        Map<Integer, List<Integer>> neighborhoods = new HashMap<>();
        for (int idx : pointIdxs) {
            Point centralPoint = imgKeyPoints[idx].pt;

            List<Integer> neighborhood =
                    pointIdxs.stream()
                            .filter(indx -> indx != idx)
                            .sorted((kp1, kp2) -> {
                                double distToCenterKp1 = DistanceOfPoint(centralPoint, imgKeyPoints[kp1].pt);
                                double distToCenterKp2 = DistanceOfPoint(centralPoint, imgKeyPoints[kp2].pt);
                                if (distToCenterKp1 > distToCenterKp2) {
                                    return -1;
                                } else if (distToCenterKp1 < distToCenterKp2) {
                                    return 1;
                                } else {
                                    return 0;
                                }
                            })
                            .limit(NumOfKeyPairs)
                            .collect(Collectors.toList());
            neighborhoods.put(idx, neighborhood);
        }

        return neighborhoods;
    }

    private double DistanceOfPoint(Point lhs, Point rhs) {
        return Math.sqrt(Math.pow(Math.abs(lhs.x - rhs.x), 2) + Math.pow(Math.abs(lhs.y - rhs.y), 2));
    }
}
