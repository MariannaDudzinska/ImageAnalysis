import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.opencv.core.Point;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.KeyPoint;

import javax.media.jai.PerspectiveTransform;
import java.awt.geom.AffineTransform;
import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class RansacFilter {
    private float MaxError = 0.01f;
    private int Iterations = 100;

    public List<DMatch> FilterRANSAC(List<DMatch> matches, KeyPoint[] firstImgKeyPoints, KeyPoint[] secondImgKeyPoints,
                                     Object transform, TransformType transformType) {

        LinkedList<DMatch> goodMatchesList = new LinkedList<>();
        for (DMatch match : matches) {
            if (IsPointGood(match, firstImgKeyPoints, secondImgKeyPoints, transform, transformType)) {
                goodMatchesList.add(match);
            }
        }
        return goodMatchesList;
    }

    public Object CalculateBestTransform(List<DMatch> matches, KeyPoint[] firstImgKeyPoints, KeyPoint[] secondImgKeyPoints,
                                         TransformType transformType) {
        Object bestModel = null;
        int bestScore = 0;
        for (int i = 0; i < Iterations; i++) {
            Object model = null;
            while (model == null) {
                List<DMatch> samples = chooseSamples(matches, transformType);
                if (transformType == TransformType.Affine) {
                    model = calculateAffineTransform(samples, firstImgKeyPoints, secondImgKeyPoints);
                } else {
                    model = calculatePerspectiveTransform(samples, firstImgKeyPoints, secondImgKeyPoints);
                }
            }
            int score = 0;
            for (DMatch match : matches) {
                if (IsPointGood(match, firstImgKeyPoints, secondImgKeyPoints, model, transformType)) {
                    score++;
                }
            }
            if (score > bestScore) {
                bestScore = score;
                bestModel = model;
            }
        }
        return bestModel;
    }

    private AffineTransform calculateAffineTransform(List<DMatch> samples, KeyPoint[] firstImgKeyPoints, KeyPoint[] secondImgKeyPoints) {
        Point firstLeft = firstImgKeyPoints[samples.get(0).queryIdx].pt;
        Point secondLeft = firstImgKeyPoints[samples.get(1).queryIdx].pt;
        Point thirdLeft = firstImgKeyPoints[samples.get(2).queryIdx].pt;
        Point firstRight = secondImgKeyPoints[samples.get(0).trainIdx].pt;
        Point secondRight = secondImgKeyPoints[samples.get(1).trainIdx].pt;
        Point thirdRight = secondImgKeyPoints[samples.get(2).trainIdx].pt;

        if (samples.size() != 3) {
            throw new IllegalArgumentException("Incorrect amount of samples");
        }
        double[][] arrayFirstMatrix = {
                {firstLeft.x, firstLeft.y, 1, 0, 0, 0},
                {secondLeft.x, secondLeft.y, 1, 0, 0, 0},
                {thirdLeft.x, thirdLeft.y, 1, 0, 0, 0},
                {0, 0, 0, firstLeft.x, firstLeft.y, 1},
                {0, 0, 0, secondLeft.x, secondLeft.y, 1},
                {0, 0, 0, thirdLeft.x, thirdLeft.y, 1}
        };
        RealMatrix firstMatrix = MatrixUtils.createRealMatrix(arrayFirstMatrix);
        if (new LUDecomposition(firstMatrix).getDeterminant() == 0) {
            return null;
        }

        RealMatrix inverseOfFirstMatrix = new LUDecomposition(firstMatrix).getSolver().getInverse();
        double[][] temporarySecondMatrix = {
                {firstRight.x}, {secondRight.x}, {thirdRight.x},
                {firstRight.y}, {secondRight.y}, {thirdRight.y}
        };
        RealMatrix secondMatrix = MatrixUtils.createRealMatrix(temporarySecondMatrix);
        double[] affineTransformParameters = inverseOfFirstMatrix.multiply(secondMatrix).getColumn(0);

        return new AffineTransform(affineTransformParameters[0], affineTransformParameters[3],
                                    affineTransformParameters[1], affineTransformParameters[4],
                                    affineTransformParameters[2], affineTransformParameters[5]);
    }

    private PerspectiveTransform calculatePerspectiveTransform(List<DMatch> samples, KeyPoint[] firstImgKeyPoints, KeyPoint[] secondImgKeyPoints) {

        Point firstLeft = firstImgKeyPoints[samples.get(0).queryIdx].pt;
        Point secondLeft = firstImgKeyPoints[samples.get(1).queryIdx].pt;
        Point thirdLeft = firstImgKeyPoints[samples.get(2).queryIdx].pt;
        Point fourthLeft = firstImgKeyPoints[samples.get(3).queryIdx].pt;
        Point firstRight = secondImgKeyPoints[samples.get(0).trainIdx].pt;
        Point secondRight = secondImgKeyPoints[samples.get(1).trainIdx].pt;
        Point thirdRight = secondImgKeyPoints[samples.get(2).trainIdx].pt;
        Point fourthRight = secondImgKeyPoints[samples.get(3).trainIdx].pt;
        if (samples.size() != 4) {
            throw new IllegalArgumentException("Wrong number of samples");
        }

        double[][] arrayFirstMatrix = {
                {firstLeft.x, firstLeft.y, 1, 0, 0, 0, -firstRight.x * firstLeft.x, -firstRight.x * firstLeft.y},
                {secondLeft.x, secondLeft.y, 1, 0, 0, 0, -secondRight.x * secondLeft.x, -secondRight.x * secondLeft.y},
                {thirdLeft.x, thirdLeft.y, 1, 0, 0, 0, -thirdRight.x * thirdLeft.x, -thirdRight.x * thirdLeft.y},
                {fourthLeft.x, fourthLeft.y, 1, 0, 0, 0, -fourthRight.x * fourthLeft.x, -fourthRight.x * fourthLeft.y},
                {0, 0, 0, firstLeft.x, firstLeft.y, 1, -firstRight.y * firstLeft.x, -firstRight.y * firstLeft.y},
                {0, 0, 0, secondLeft.x, secondLeft.y, 1, -secondRight.y * secondLeft.x, -secondRight.y * secondLeft.y},
                {0, 0, 0, thirdLeft.x, thirdLeft.y, 1, -thirdRight.y * thirdLeft.x, -thirdRight.y * thirdLeft.y},
                {0, 0, 0, fourthLeft.x, fourthLeft.y, 1, -fourthRight.y * fourthLeft.x, -fourthRight.y * fourthLeft.y}
        };

        RealMatrix firstMatrix = MatrixUtils.createRealMatrix(arrayFirstMatrix);

        if (new LUDecomposition(firstMatrix).getDeterminant() == 0) {
            return null;
        }

        RealMatrix inverseOfFirstMatrix = new LUDecomposition(firstMatrix).getSolver().getInverse();

        double[][] temporarySecondMatrix = {
                {firstRight.x}, {secondRight.x}, {thirdRight.x}, {fourthRight.x},
                {firstRight.y}, {secondRight.y}, {thirdRight.y}, {fourthRight.y},
        };

        RealMatrix secondMatrix = MatrixUtils.createRealMatrix(temporarySecondMatrix);

        double[] affineTransformParameters = inverseOfFirstMatrix.multiply(secondMatrix).getColumn(0);
        double[][] twoDimentionalAffineTransformParameters = {{affineTransformParameters[0], affineTransformParameters[1], affineTransformParameters[2]},
                                                                {affineTransformParameters[3], affineTransformParameters[4], affineTransformParameters[5]},
                                                                {affineTransformParameters[6], affineTransformParameters[7], 1}};

        return new PerspectiveTransform(twoDimentionalAffineTransformParameters);
    }

    
    private List<DMatch> chooseSamples(List<DMatch> matches, TransformType transformType) {
        if (transformType == TransformType.Affine) {
            if (matches.size() < 3) {
                throw new IllegalArgumentException("Not enough pairs to assemble a sample");
            }
        } else {
            if (matches.size() < 4) {
                throw new IllegalArgumentException("Not enough pairs to assemble a sample");
            }
        }
        List<DMatch> samples = new ArrayList<>();
        Random random = new Random();

        int maximumSize = transformType == TransformType.Affine ? 3 : 4;

        while (samples.size() < maximumSize) {
            DMatch pair = matches.get(random.nextInt(matches.size()));
            if (!samples.contains(pair)) {
                samples.add(pair);
            }
        }
        return samples;
    }

    private boolean IsPointGood(DMatch match, KeyPoint[] firstImgKeyPoints, KeyPoint[] secondImgKeyPoints, Object transform, TransformType transformType) {
        KeyPoint startKpt = firstImgKeyPoints[match.queryIdx];
        KeyPoint endKpt = secondImgKeyPoints[match.trainIdx];

        Point2D start = new Point2D.Double(startKpt.pt.x, startKpt.pt.y);
        Point2D end = new Point2D.Double(endKpt.pt.x, endKpt.pt.y);
        Point2D expectedAftrTransform;

        if (transformType == TransformType.Affine) {
            expectedAftrTransform = ((AffineTransform) transform).transform(start, new Point2D.Double());
        } else {
            expectedAftrTransform = ((PerspectiveTransform) transform).transform(start, new Point2D.Double());
        }
        assert expectedAftrTransform != null;

        double error = expectedAftrTransform.distance(end);

        return error <= MaxError;
    }
}
