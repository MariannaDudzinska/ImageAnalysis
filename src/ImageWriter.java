import org.opencv.core.*;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.Highgui;

import java.util.List;

public class ImageWriter {
    private static Scalar keyPointColor = new Scalar(255, 0, 0);
    private static Scalar matchColor = new Scalar(0, 255, 0);

    public static void WriteImg(Mat srcImage, MatOfKeyPoint keyPoints, String path) {
        Mat outputImage = new Mat(srcImage.rows(), srcImage.cols(), Highgui.CV_LOAD_IMAGE_COLOR);

        try {
            Features2d.drawKeypoints(srcImage, keyPoints, outputImage, keyPointColor, 0);
            Highgui.imwrite(path, outputImage);
        } catch (Exception e) {
            System.out.println("Exception occured when writing image: " + e.toString());
        }
    }

    public static void WriteMatchesForTwoImages(
            Mat firstImage,
            MatOfKeyPoint firstImageKeyPoints,
            Mat secondImage,
            MatOfKeyPoint secondImageKeypoints,
            List<DMatch> goodMatchesList,
            String outputPath) {

        Mat matchesOutputImage = new Mat(firstImage.rows() * 2, firstImage.cols() * 2, Highgui.CV_LOAD_IMAGE_COLOR);

        try {
            MatOfDMatch goodMatches = new MatOfDMatch();
            goodMatches.fromList(goodMatchesList);

            Features2d.drawMatches(firstImage,
                    firstImageKeyPoints,
                    secondImage,
                    secondImageKeypoints,
                    goodMatches,
                    matchesOutputImage,
                    matchColor,
                    keyPointColor,
                    new MatOfByte(),
                    2
            );
            Highgui.imwrite(outputPath, matchesOutputImage);
        } catch (Exception e) {
            System.out.println("Exception occured when writing matches of two images: " + e.toString());
        }
    }


}
