function nmse = NMSE(Xhat,X)

err = (Xhat(:)-X(:));
gt = X(:);
nmse = 10*log10(sum(err.*conj(err))/sum(gt.*conj(gt)));

end

