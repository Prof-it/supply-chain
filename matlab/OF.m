function Cost = OF(x, y, E, C, Cp, Cpp, d, gama)
[n_candidate, n_types] = size(x);
[nc, n_community] = size(y);

% Total construction cost
cost_build = sum(sum(x .* C));  % C is 1×3, x is n×3

% Distance cost
cost_travel = sum(sum(Cp * (d .* y)));

% Penalty cost
cost_penalty = sum(sum(Cpp * (gama .* y)));

% Total cost
f1 = cost_build + cost_travel + cost_penalty;

% Fairness: std dev of assigned distances
distances = [];
for i = 1:n_candidate
    for j = 1:n_community
        if y(i,j) > 0
            distances = [distances; repmat(d(i,j), y(i,j), 1)];
        end
    end
end
if length(distances) > 1
    f2 = std(distances);
else
    f2 = 0;  % If only one allocation, std is zero
end

Cost = [f1; f2];
end
