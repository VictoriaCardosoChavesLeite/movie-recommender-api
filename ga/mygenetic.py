

from ga.algorithm import Algorithm
from sqlalchemy.orm import Session
from fastapi import Depends
import numpy as np
import math 

from db.database import get_db
from db.repositories import UserRepository, MovieRepository, RatingsRepository

class MyGeneticAlgorithm(Algorithm):

    def __init__(self, query_search, individual_size, population_size, p_crossover, p_mutation, all_ids, max_generations=100, size_hall_of_fame=1, fitness_weights=(1.0, ), seed=42, db=None) -> None:


        super().__init__(
            individual_size, 
            population_size, 
            p_crossover, 
            p_mutation, 
            all_ids, 
            max_generations, 
            size_hall_of_fame, 
            fitness_weights, 
            seed)
        
        self.db = db
        self.all_ids = all_ids
        self.query_search = query_search
        

    
    def evaluate(self, individual):
        
        if len(individual) != len(set(individual)):
            return (0.0, )
        
        if len(list(set(individual) - set(self.all_ids))) > 0:
            return (0.0, )
        
        ###################################################
        #Pegar os gêneros favoritos do usuário
        user_ratings = RatingsRepository.find_by_userid(self.db, self.query_search) #Filmes que o usuário avaliou
        user_movies_ids = [rating.movieId for rating in user_ratings] #Pega os IDs dos filmes que o usuário avaliou e armazena em uma lista
        user_movies = MovieRepository.find_all_ids(self.db,user_movies_ids) #Pegar as informações dos filmes

        user_genres = set() #Coleção de objetos onde serão armazenados os gêneros
        for movie in user_movies:
            genres = movie.genres.split("|")#Pega o genêro do filme tirando as barras
            user_genres.update(genres)#Armazena o gênero

        #Pegar os gêneros dos filmes que serão recomendados para o usuário, a lógica é a mesma de pegar os gêneros favoritos do usuário
        recommend_movies = MovieRepository.find_all_ids(self.db,individual)
        recommend_genres = set()
        for movie in recommend_movies:
            genres = movie.genres.split("|")
            recommend_genres.update(genres)
        ###################################################

        # Fetch ratings for recommended movies
        ratings_movies = RatingsRepository.find_by_movieid_list(self.db, individual)

        if len(ratings_movies) > 0:
            # Calculate the average rating of recommended movies
            mean_rating = np.mean([obj_.rating for obj_ in ratings_movies])
        else:
            # If no ratings are found, assign a low fitness score
            mean_rating = 0.0

        # Define a threshold for a "good" recommendation (you can adjust this)
        good_recommendation_threshold = 3.5

        # Calculate the fitness score based on the average rating
        # A higher score indicates a better recommendation
        fitness_score = mean_rating / good_recommendation_threshold

        # Ensure the fitness score is within the range [0, 1]
        fitness_score = max(0.0, min(fitness_score, 1.0))

        return (fitness_score, )

        

        

