

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
        # Pegar os gêneros favoritos do usuário
        user_ratings = RatingsRepository.find_by_userid(self.db, self.query_search)  # Filmes que o usuário avaliou
        user_movies_ids = [rating.movieId for rating in user_ratings]  # Pega os IDs dos filmes que o usuário avaliou e armazena em uma lista
        user_movies = MovieRepository.find_all_ids(self.db, user_movies_ids)  # Pegar as informações dos filmes

        # Criar um dicionário que mapeia os IDs dos filmes às notas do usuário
        user_ratings_dict = {rating.movieId: rating.rating for rating in user_ratings}

        user_genres = set()
        for movie in user_movies:
            rating = user_ratings_dict.get(movie.movieId)
            if rating >= 3.7:
                genres = movie.genres.split("|")
                user_genres.update(genres)

        # Pegar os gêneros dos filmes que serão recomendados para o usuário, a lógica é a mesma de pegar os gêneros favoritos do usuário
        recommend_movies = MovieRepository.find_all_ids(self.db, individual)
        recommend_genres = set()
        for movie in recommend_movies:
            genres = movie.genres.split("|")
            recommend_genres.update(genres)

        # Verificar se há interseção entre os gêneros favoritos do usuário e os gêneros dos filmes recomendados
        intersection_genres = user_genres.intersection(recommend_genres)

        if intersection_genres:
            # Se houver interseção, calcular a média das classificações apenas para os filmes que têm gêneros favoritos
            ratings_movies = RatingsRepository.find_by_movieid_list(self.db, individual)
            mean_rating = 0.0
            num_ratings = 0

            for obj_ in ratings_movies:
                if obj_.movieId in user_movies_ids:
                    # Obter a nota do usuário para o filme atual
                    user_rating = user_ratings_dict.get(obj_.movieId, 0.0)
                    mean_rating += user_rating
                    num_ratings += 1

            if num_ratings > 0:
                mean_rating /= num_ratings
            else:
                mean_rating = 0.0

            # Ajustar o limite para uma "boa" recomendação
            good_recommendation_threshold = 3.7

            # Calcular a pontuação de fitness com base na média das classificações
            # Uma pontuação mais alta indica uma melhor recomendação
            fitness_score = (mean_rating / good_recommendation_threshold) + 0.5

            # Garantir que a pontuação de fitness esteja dentro do intervalo [0, 1]
            fitness_score = max(0.0, min(fitness_score, 1.0))
        else:
            # Se não houver interseção, calcular a média das classificações para todos os filmes recomendados
            ratings_movies = RatingsRepository.find_by_movieid_list(self.db, individual)
            mean_rating = 0.0
            num_ratings = len(ratings_movies)

            if num_ratings > 0:
                mean_rating = np.mean([obj_.rating for obj_ in ratings_movies])
            else:
                mean_rating = 0.0

            # Ajustar o limite para uma "boa" recomendação
            good_recommendation_threshold = 3.7

            # Calcular a pontuação de fitness com base na média das classificações
            # Uma pontuação mais alta indica uma melhor recomendação
            fitness_score = mean_rating / good_recommendation_threshold

            # Garantir que a pontuação de fitness esteja dentro do intervalo [0, 1]
            fitness_score = max(0.0, min(fitness_score, 1.0))

        return (fitness_score, )
