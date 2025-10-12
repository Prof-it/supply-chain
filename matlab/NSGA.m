clc
clear;
close all;

%% Load initial data
initial_data;

%% NSGA-II Parameters
MaxIt = 50;
nPop = 10;

pCrossover = 0.7;
nCrossover = round(pCrossover * nPop / 2) * 2;  % Must be even
pMutation = 0.15;
nMutation = round(pMutation * nPop);

n_types = length(Q);  % 3 types

%% Empty Individual Structure
empty_individual.x = zeros(n_candidate, n_types);  % x(i,j)
empty_individual.y = zeros(n_candidate, n_community);  % Allocations y(i,j)
empty_individual.E = zeros(n_community, 1);  % Fuzzy demand
empty_individual.Cost = [];
empty_individual.Rank = [];
empty_individual.DominationSet = [];
empty_individual.DominatedCount = [];
empty_individual.CrowdingDistance = [];

%% Initial Population Generation
pop = repmat(empty_individual, nPop, 1);
i = 1;
while i <= nPop
    x = zeros(n_candidate, n_types);
    selected = randperm(n_candidate);
    count = 0;
    for idx = 1:length(selected)
        if count >= U
            break;
        end
        loc = selected(idx);
        % Check D constraint
        too_close = false;
        for jdx = 1:n_candidate
            if any(x(jdx, :)) && dpp(loc, jdx) < D
                too_close = true;
                break;
            end
        end
        if too_close
            continue;
        end
        % Randomly assign a type
        t = randi(n_types);
        x(loc, t) = 1;
        count = count + 1;
    end
    pop(i).x = x;
    pop(i).E = floor(((E_U - E_L) .* rand(n_community, 1)) + E_L);
    pop(i).y = allocation(x, pop(i).E, Q, d);
    pop(i).Cost = OF(x, pop(i).y, pop(i).E, C, Cp, Cpp, d, gama);
    i = i + 1;
end

%% Non-dominated Sorting & Crowding
[pop, F] = NonDominatedSorting(pop);
pop = CalcCrowdingDistance(pop, F);
pop = SortPopulation(pop);

%% Main NSGA-II Loop
for it = 1:MaxIt

    % ----- Crossover -----
    popc = repmat(empty_individual, nCrossover, 1);
    k = 1;
    while k <= nCrossover
        i1 = randi([1 nPop]);
        i2 = randi([1 nPop]);
        if i1 == i2, continue; end
        p1 = pop(i1);
        p2 = pop(i2);
        [x_new, notpossible] = Crossover(p1, p2, Q, D, dpp, E_L, E_U);
        if notpossible == 0
            popc(k).x = x_new;
            popc(k).E = floor(((E_U - E_L) .* rand(n_community, 1)) + E_L);
            popc(k).y = allocation(x_new, popc(k).E, Q, d);
            popc(k).Cost = OF(x_new, popc(k).y, popc(k).E, C, Cp, Cpp, d, gama);
            k = k + 1;
        end
    end

    % ----- Mutation -----
    popm = repmat(empty_individual, nMutation, 1);
    for k = 1:nMutation
        i1 = randi([1 nPop]);
        p = pop(i1);
        x_new = Mutation(p.x, dpp, D);
        popm(k).x = x_new;
        popm(k).E = floor(((E_U - E_L) .* rand(n_community, 1)) + E_L);
        popm(k).y = allocation(x_new, popm(k).E, Q, d);
        popm(k).Cost = OF(x_new, popm(k).y, popm(k).E, C, Cp, Cpp, d, gama);
    end

    % ----- Merge and Select -----
    pop = [pop; popc; popm];
    [pop, F] = NonDominatedSorting(pop);
    pop = CalcCrowdingDistance(pop, F);
    pop = SortPopulation(pop);
%     pop = pop(1:nPop);

    % ----- Show Iteration Info -----
    F1 = pop(F{1});
    disp(['Iteration ', num2str(it), ' | Number of F1 Members = ', num2str(numel(F{1}))]);
    figure(1); PlotCosts(F1);
end
