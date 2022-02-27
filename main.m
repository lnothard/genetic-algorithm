load("xy.mat");

% parameters
iter = 6000;
population_size = 50;
mutation_rate = 0.2;
crossover_rate = 0.6;
tournament_size = 4;

tic;

population = zeros(population_size, 100);
f = zeros(population_size, 1);

for i = 1: population_size
    % each chromosome is a random permutation of numbers 1-100 
    population(i,:) = randperm(100); 
    % calculate the fitness of each chromosome and store in the same index as the population
    f(i) = fitness(population(i,:), xy); 
end

for i = 1: iter
    % sort the fitness from highest to lowest and store the indices
    [~,idx] = sort(f, 'descend'); 
    new_population = zeros(population_size,100);
    % elitism - retain the two fittest members of the population in the next generation
    new_population(1,:) = population(idx(1),:);
    new_population(2,:) = population(idx(2),:);
    for j = 4:2:population_size
        % probabilistically select two parents from the population - 
        % fitter chromosomes have a higher chance of being selected
        parents = tournament_selection(population, population_size, f, tournament_size);
%         parents = roulette_wheel_selection(population, population_size, f);
        
        if rand <= crossover_rate
            % create two offspring from the selected parents using the
            % chosen crossover operator
            offspring = partially_mapped_crossover(parents(1,:), parents(2,:));
%             offspring = order_based_crossover(parents(1,:), parents(2,:));
        else
            % if no crossover is performed, the offspring are simply copies
            % of the parents
            offspring = parents;
        end
        
        if rand <= mutation_rate
            % mutate each offspring
            offspring(1,:) = reverse_mutation(offspring(1,:));
            offspring(2,:) = reverse_mutation(offspring(2,:));
%             offspring(1,:) = swap_mutation(offspring(1,:));
%             offspring(2,:) = swap_mutation(offspring(2,:));
        end
        
        % accept the offspring into the new population
        new_population(j-1,:) = offspring(1,:);
        new_population(j,:) = offspring(2,:);
    end
    % replace the old population with the new
    population = new_population;

    % evaluate the fitness of the new population
    for j = 1: population_size
        f(j) = fitness(population(j,:), xy);
    end
end
% find the chromosome with the highest fitness value from the final
% generation
[max_fitness, idx] = max(f(:)); 
max_fitness
optRoute = population(idx,:);
% calculate the distance travelled using the most optimal route. since
% fitness = 1 / distance; distance = 1 / fitness
minDist = 1 ./ max_fitness;

% plot the most optimal route
figure('Name','TSP_GA | Results','Numbertitle','off');
subplot(2,2,1);
pclr = ~get(0,'DefaultAxesColor');
plot(xy(:,1),xy(:,2),'.','Color',pclr);
title('City Locations');
subplot(2,2,2);
rte = optRoute([1:100 1]);
plot(xy(rte,1),xy(rte,2),'r.-');
title(sprintf('Total Distance = %1.4f',minDist));

% print the total running time of the GA
toc

% fitness function
function f = fitness(chromosome, xy)
    % rearrange the city coordinates in the order of the chromosome whose
    % fitness if being evaluated
    A = xy(chromosome,:);
    % calculate the distances between each coordinate from start to end
    diff_ = [diff(A,1); A(end,:)-A(1,:)];
    dist = sqrt(sum(diff_ .* diff_, 2));
    % also calculate the distance from the end back to the start
    return_dist = sqrt((A(1,1) - A(100,1))^2 + (A(1,2) - A(100,2))^2);
    % take the sum of the distances
    total_dist = sum(dist) + return_dist;
    % fitness is inversely proportional to the total distance
    f = 1 ./ total_dist;
end

function parents = tournament_selection(population, population_size, f, tournament_size)
    parents = zeros(2, 100);
    best = 0;
    for i = 1: 2
        % choose [tournament size] random chromosomes from the population
        rand_sample = randsample(population_size, tournament_size);
        % pick the chromosome with the highest fitness
        for j = 1: tournament_size
            if f(rand_sample(j)) > best
                best = f(rand_sample(j));
                idx = rand_sample(j);
            end
        end
        % the parent is the fittest chromosome in the tournament
        parents(i,:) = population(idx,:);
    end
end

function parents = roulette_wheel_selection(population, population_size, f)
    parents = zeros(2, 100);
    % calculate the sum of the fitness values to get the total fitness
    total_fitness = sum(f);
    cumulative_probability = zeros(population_size, 1);
    % calculate probability of a chromosome by dividing its fitness by the
    % total fitness
    cumulative_probability(1) = f(1) / total_fitness;
    for i = 2: population_size
        cumulative_probability(i) = cumulative_probability(i-1) + (f(i) / total_fitness);
    end
    for j = 1: 2
        % generate random number between 0 and 1
        uniform_random = rand;
        % the selected parent is the one with the closest cumulative
        % probability value to the uniform random number. This way
        % chromosomes with higher fitness value will have a higher
        % probability of selection
        [~, idx] = min(abs(cumulative_probability - uniform_random));
        parents(j,:) = population(idx,:);
    end
end

function offspring = partially_mapped_crossover(parent1, parent2)
    offspring = zeros(2, 100);
    % get two random crossover points
    crossover_points = sort(randsample(100, 2));
    % initialize offspring to be same as parents
    offspring(1,:) = parent1;
    offspring(2,:) = parent2;
    
    % at each index between the crossover points, the value of the first
    % offspring becomes the value of the second parent at that index, and
    % the original value at this index in the offspring swaps position with
    % the value that is replacing it. Repeat for the second offspring with
    % the roles of the parents reversed
    for i = crossover_points(1): crossover_points(2)
        idx = offspring(1,:) == parent2(i);
        offspring(1,idx) = offspring(1,i);
        offspring(1,i) = parent2(i);
        
        idx = offspring(2,:) == parent1(i);
        offspring(2,idx) = offspring(2,i);
        offspring(2,i) = parent1(i);
    end
end

function offspring = order_based_crossover(parent1, parent2)
    offspring = zeros(2,100);
    offspring(1,:) = parent1;
    offspring(2,:) = parent2;
    
    % generate 4 random points
    pos = randsample(100, 4);
    % take the values at these points in the second parent
    % find the positions where these values are located in the first
    % parent
    [~,idx] = ismember(parent2(pos), parent1);
    % insert the values in the order they appear in the second parent
    offspring(1,sort(idx)) = parent2(pos);
    
    % repeat with roles of parents reversed
    [~,idx] = ismember(parent1(pos), parent2);
    offspring(2,sort(idx)) = parent1(pos);
end

function chromosome = swap_mutation(chromosome)
    % swap the values at two random positions
    rand_sample = randsample(100, 2);
    temp = chromosome(rand_sample(1));
    chromosome(rand_sample(1)) = chromosome(rand_sample(2));
    chromosome(rand_sample(2)) = temp;
end

function chromosome = reverse_mutation(chromosome)
    % reverse the order of the values between two random positions
    rand_sample = sort(randsample(100, 2));
    chromosome(rand_sample(1) : rand_sample(2)) = flip(chromosome(rand_sample(1) : rand_sample(2)));
end