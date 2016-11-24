function error = CalculateError(Yval, Yest, isRegression)
    N = size(Yval, 1);

    if (isRegression == 1)
        temp = (Yval - Yest) .^ 2;
        error = sum(temp) / (2 * N);
    else
        temp = (Yval ~= Yest);
        error = mean(temp);
    end