import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;

import java.util.List;


public class Main {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        ImageSimilarity imageSimiliarity = new ImageSimilarity();

        Mat firstImage = Highgui.imread("C:\\Users\\Admin\\Desktop\\SI 4\\img\\eggo1.jpg", Highgui.CV_LOAD_IMAGE_COLOR);
        Mat secondImage = Highgui.imread("C:\\Users\\Admin\\Desktop\\SI 4\\img\\eggo2.jpg", Highgui.CV_LOAD_IMAGE_COLOR);

        MatOfKeyPoint firstImageKeyPoints = imageSimiliarity.ExtractKeyPoints(firstImage);
        MatOfKeyPoint firstImageKeyPointsDescriptors = imageSimiliarity.ExtractDescriptors(firstImage, firstImageKeyPoints);

        ImageWriter.WriteImg(firstImage, firstImageKeyPoints, "C:\\Users\\Admin\\Desktop\\SI 4\\obrazki\\2firstImageKeyPoints.jpg");

        MatOfKeyPoint secondImageKeyPoints = imageSimiliarity.ExtractKeyPoints(secondImage);
        MatOfKeyPoint secondImageKeyPointsDescriptors = imageSimiliarity.ExtractDescriptors(secondImage, secondImageKeyPoints);

        ImageWriter.WriteImg(secondImage, secondImageKeyPoints, "C:\\Users\\Admin\\Desktop\\SI 4\\obrazki\\2secondImageKeyPoints.jpg");

        List<DMatch> keyPointPairs = imageSimiliarity.KeyPointPairsCreation(firstImageKeyPointsDescriptors, secondImageKeyPointsDescriptors);

        KeyPoint[] firstImageKeyPointsArray = firstImageKeyPoints.toArray();
        KeyPoint[] secondImageKeyPointsArray = secondImageKeyPoints.toArray();


        List<DMatch> filteredKeyPointPairs = imageSimiliarity.FilterMatchedKeyPoints(keyPointPairs, firstImageKeyPointsArray, secondImageKeyPointsArray);

        ImageWriter.WriteMatchesForTwoImages(firstImage, firstImageKeyPoints, secondImage, secondImageKeyPoints, filteredKeyPointPairs,
                "C:\\Users\\Admin\\Desktop\\SI 4\\obrazki\\3twoImagesMatches.jpg");
        System.out.println(filteredKeyPointPairs.size());

        RansacFilter ransac = new RansacFilter();
        Object bestAffineTransform =
               ransac.CalculateBestTransform(filteredKeyPointPairs, firstImageKeyPointsArray, secondImageKeyPointsArray, TransformType.Affine);
        Object bestPerspectiveTransform =
                ransac.CalculateBestTransform(filteredKeyPointPairs, firstImageKeyPointsArray, secondImageKeyPointsArray, TransformType.Perspective);

        List<DMatch> ransacFilteredKeyPointPairs = ransac.FilterRANSAC(filteredKeyPointPairs, firstImageKeyPointsArray, secondImageKeyPointsArray, bestAffineTransform, TransformType.Affine);

        ImageWriter.WriteMatchesForTwoImages(firstImage, firstImageKeyPoints, secondImage, secondImageKeyPoints, ransacFilteredKeyPointPairs,
                "C:\\Users\\Admin\\Desktop\\SI 4\\obrazki\\4ransacTwoImagesMatches.jpg");
    }
}