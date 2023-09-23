from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship

from db.database import Base 

class Movie(Base):

    __tablename__  = "movies"

    movieId: int = Column(Integer, primary_key=True, index=True)
    title: str = Column(String(200), nullable=False)
    year: int = Column(Integer, nullable=True)
    genres: str = Column(String(200), nullable=True)
    url_poster: str = Column(String(300), nullable=True)
    imdbId: str = Column(String(10), nullable=True)

    def to_dict(self):
        return {
            "movieId": self.movieId,
            "title": self.title,
            "year": self.year,
            "genres":self.genres,
            "url_poster":self.url_poster,
            "imdbId":self.imdbId
        }


class Ratings(Base):

    __tablename__ = "ratings"

    userId: int = Column(Integer, ForeignKey("users.userId"), primary_key=True)
    movieId: int = Column(Integer, ForeignKey("movies.movieId"), primary_key=True)
    rating: float = Column(Float, nullable=False)

    movie = relationship("Movie")
    user = relationship("User")

class User(Base):
    
    __tablename__ = "users"

    userId: int = Column(Integer, primary_key=True, index=True)
    userName: str = Column(String(50), nullable=False)