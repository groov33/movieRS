"""
Movie Recommendation Dataset Generator
Generates curated movies metadata with directors, cast, genres, plot overviews,
and simulated realistic user ratings across different user archetypes.
"""

import os
import pandas as pd
import numpy as np
import random

MOVIES_DATA = [
    # Sci-Fi / Space / Nolan
    {"movie_id": 1, "title": "Inception", "genres": "Action|Sci-Fi|Thriller", "director": "Christopher Nolan", 
     "cast": "Leonardo DiCaprio|Joseph Gordon-Levitt|Elliot Page|Tom Hardy", "release_year": 2010, "vote_average": 8.8,
     "overview": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.",
     "keywords": "subconscious dreams heist reality manipulation corporate espionage mind-bending", "poster": "https://images.unsplash.com/photo-1534447677768-be436bb09401?w=500&auto=format&fit=crop"},
    
    {"movie_id": 2, "title": "Interstellar", "genres": "Adventure|Drama|Sci-Fi", "director": "Christopher Nolan",
     "cast": "Matthew McConaughey|Anne Hathaway|Jessica Chastain|Michael Caine", "release_year": 2014, "vote_average": 8.7,
     "overview": "When Earth becomes uninhabitable in the future, a farmer and ex-NASA pilot, Joseph Cooper, is tasked to pilot a spacecraft, along with a team of researchers, to find a new planet for humans.",
     "keywords": "black hole wormhole time dilation relativity space exploration father daughter love humanity survival", "poster": "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=500&auto=format&fit=crop"},
     
    {"movie_id": 3, "title": "The Dark Knight", "genres": "Action|Crime|Drama|Thriller", "director": "Christopher Nolan",
     "cast": "Christian Bale|Heath Ledger|Aaron Eckhart|Michael Caine", "release_year": 2008, "vote_average": 9.0,
     "overview": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
     "keywords": "gotham superhero joker vigilante psychological warfare corruption chaos morality", "poster": "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=500&auto=format&fit=crop"},

    {"movie_id": 4, "title": "The Prestige", "genres": "Drama|Mystery|Sci-Fi|Thriller", "director": "Christopher Nolan",
     "cast": "Christian Bale|Hugh Jackman|Scarlett Johansson|Michael Caine", "release_year": 2006, "vote_average": 8.5,
     "overview": "After a tragic accident, two stage magicians in 1890s London engage in a battle to create the ultimate illusion while sacrificing everything they have to outwit each other.",
     "keywords": "magicians rivalry obsession teleportation illusion twist ending science victorian", "poster": "https://images.unsplash.com/photo-1514533450685-4493e01d1fdc?w=500&auto=format&fit=crop"},

    {"movie_id": 5, "title": "Tenet", "genres": "Action|Sci-Fi|Thriller", "director": "Christopher Nolan",
     "cast": "John David Washington|Robert Pattinson|Elizabeth Debicki|Kenneth Branagh", "release_year": 2020, "vote_average": 7.3,
     "overview": "Armed with only one word, Tenet, and fighting for the survival of the entire world, a Protagonist journeys through a twilight world of international espionage on a mission that unfolds in something beyond real time.",
     "keywords": "time inversion entropy espionage physics paradox temporal cold war", "poster": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=500&auto=format&fit=crop"},

    {"movie_id": 6, "title": "The Matrix", "genres": "Action|Sci-Fi", "director": "Lana Wachowski|Lilly Wachowski",
     "cast": "Keanu Reeves|Laurence Fishburne|Carrie-Anne Moss|Hugo Weaving", "release_year": 1999, "vote_average": 8.7,
     "overview": "When a beautiful stranger leads computer hacker Neo to a forbidding underworld, he discovers the shocking truth--the life he knows is the elaborate deception of an evil cyber-intelligence.",
     "keywords": "simulation artificial intelligence cyberpunk hacker red pill chosen one martial arts dystopia", "poster": "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=500&auto=format&fit=crop"},

    {"movie_id": 7, "title": "Blade Runner 2049", "genres": "Action|Drama|Mystery|Sci-Fi", "director": "Denis Villeneuve",
     "cast": "Ryan Gosling|Harrison Ford|Ana de Armas|Sylvia Hoeks", "release_year": 2017, "vote_average": 8.0,
     "overview": "Young Blade Runner K's discovery of a long-buried secret leads him to track down former Blade Runner Rick Deckard, who's been missing for thirty years.",
     "keywords": "replicant cyberpunk neo-noir dystopia artificial soul detective neon atmospheric", "poster": "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=500&auto=format&fit=crop"},

    {"movie_id": 8, "title": "Dune", "genres": "Action|Adventure|Drama|Sci-Fi", "director": "Denis Villeneuve",
     "cast": "Timothée Chalamet|Rebecca Ferguson|Zendaya|Oscar Isaac", "release_year": 2021, "vote_average": 8.1,
     "overview": "A noble family becomes embroiled in a war for control over the galaxy's most valuable asset while its heir becomes troubled by visions of a dark future.",
     "keywords": "desert spice planet prophecy sandworm empire interstellar politics messiah", "poster": "https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?w=500&auto=format&fit=crop"},

    {"movie_id": 9, "title": "Arrival", "genres": "Drama|Mystery|Sci-Fi", "director": "Denis Villeneuve",
     "cast": "Amy Adams|Jeremy Renner|Forest Whitaker|Michael Stuhlbarg", "release_year": 2016, "vote_average": 7.9,
     "overview": "A linguist is recruited by the military to assist in translating alien communications after twelve mysterious spacecraft appear around the world.",
     "keywords": "first contact linguistics non-linear time extraterrestrial communication memory fate", "poster": "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?w=500&auto=format&fit=crop"},

    # Crime / Tarantino / Scorsese
    {"movie_id": 10, "title": "Pulp Fiction", "genres": "Crime|Drama", "director": "Quentin Tarantino",
     "cast": "John Travolta|Samuel L. Jackson|Uma Thurman|Bruce Willis", "release_year": 1994, "vote_average": 8.9,
     "overview": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
     "keywords": "nonlinear hitmen gangster briefcase overdose monologue dark humor redemption pop culture", "poster": "https://images.unsplash.com/photo-1485846234645-a62644f84728?w=500&auto=format&fit=crop"},

    {"movie_id": 11, "title": "Inglourious Basterds", "genres": "Adventure|Drama|War", "director": "Quentin Tarantino",
     "cast": "Brad Pitt|Christoph Waltz|Mélanie Laurent|Michael Fassbender", "release_year": 2009, "vote_average": 8.4,
     "overview": "In Nazi-occupied France during World War II, a plan to assassinate Nazi leaders by a group of Jewish U.S. soldiers coincides with a theatre owner's vengeful plans for the same.",
     "keywords": "world war ii alternate history revenge cinema resistance nazi hunter suspense", "poster": "https://images.unsplash.com/photo-1579965342575-16428a7c8881?w=500&auto=format&fit=crop"},

    {"movie_id": 12, "title": "Django Unchained", "genres": "Drama|Western", "director": "Quentin Tarantino",
     "cast": "Jamie Foxx|Christoph Waltz|Leonardo DiCaprio|Kerry Washington", "release_year": 2012, "vote_average": 8.5,
     "overview": "With the help of a German bounty-hunter, a freed slave sets out to rescue his wife from a brutal plantation owner in Mississippi.",
     "keywords": "bounty hunter western revenge slavery plantation shootout quick draw justice", "poster": "https://images.unsplash.com/photo-1508739773434-c26b3d09e071?w=500&auto=format&fit=crop"},

    {"movie_id": 13, "title": "Goodfellas", "genres": "Biography|Crime|Drama", "director": "Martin Scorsese",
     "cast": "Robert De Niro|Ray Liotta|Joe Pesci|Lorraine Bracco", "release_year": 1990, "vote_average": 8.7,
     "overview": "The story of Henry Hill and his life in the mafia, covering his relationship with his wife Karen and his mob partners Jimmy Conway and Tommy DeVito.",
     "keywords": "mafia organized crime wise guy heist drugs betrayal rise and fall narration brooklyn", "poster": "https://images.unsplash.com/photo-1542204165-65bf26472b9b?w=500&auto=format&fit=crop"},

    {"movie_id": 14, "title": "The Departed", "genres": "Crime|Drama|Thriller", "director": "Martin Scorsese",
     "cast": "Leonardo DiCaprio|Matt Damon|Jack Nicholson|Mark Wahlberg", "release_year": 2006, "vote_average": 8.5,
     "overview": "An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang in South Boston.",
     "keywords": "undercover mole irish mob police corruption boston double agent betrayal cat and mouse", "poster": "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?w=500&auto=format&fit=crop"},

    {"movie_id": 15, "title": "The Wolf of Wall Street", "genres": "Biography|Comedy|Crime|Drama", "director": "Martin Scorsese",
     "cast": "Leonardo DiCaprio|Jonah Hill|Margot Robbie|Matthew McConaughey", "release_year": 2013, "vote_average": 8.2,
     "overview": "Based on the true story of Jordan Belfort, from his rise to a wealthy stock-broker living the high life to his fall involving crime, corruption and the federal government.",
     "keywords": "wall street stockbroker greed hedonism fraud fbi wild parties excess wealth", "poster": "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=500&auto=format&fit=crop"},

    {"movie_id": 16, "title": "Fight Club", "genres": "Drama", "director": "David Fincher",
     "cast": "Brad Pitt|Edward Norton|Meat Loaf|Helena Bonham Carter", "release_year": 1999, "vote_average": 8.8,
     "overview": "An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into much more.",
     "keywords": "insomnia alter ego consumerism anarchy underground club psychological twist identity", "poster": "https://images.unsplash.com/photo-1517849845537-4d257902454a?w=500&auto=format&fit=crop"},

    {"movie_id": 17, "title": "Se7en", "genres": "Crime|Drama|Mystery|Thriller", "director": "David Fincher",
     "cast": "Morgan Freeman|Brad Pitt|Kevin Spacey|Gwyneth Paltrow", "release_year": 1995, "vote_average": 8.6,
     "overview": "Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives.",
     "keywords": "serial killer seven deadly sins dark rain grim detective box twist ending gloomy", "poster": "https://images.unsplash.com/photo-1478760329108-5c3ed9d495a0?w=500&auto=format&fit=crop"},

    {"movie_id": 18, "title": "Zodiac", "genres": "Crime|Drama|Mystery|Thriller", "director": "David Fincher",
     "cast": "Jake Gyllenhaal|Mark Ruffalo|Robert Downey Jr.|Anthony Edwards", "release_year": 2007, "vote_average": 7.7,
     "overview": "Between 1968 and 1983, a San Francisco cartoonist becomes an amateur detective obsessed with tracking down the Zodiac Killer, an unidentified murderer who terrorizes northern California.",
     "keywords": "zodiac killer journalist obsession newspaper cipher cold case investigative", "poster": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=500&auto=format&fit=crop"},

    # Romance / Drama
    {"movie_id": 19, "title": "La La Land", "genres": "Comedy|Drama|Music|Romance", "director": "Damien Chazelle",
     "cast": "Ryan Gosling|Emma Stone|Rosemarie DeWitt|J.K. Simmons", "release_year": 2016, "vote_average": 8.0,
     "overview": "While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations for the future.",
     "keywords": "jazz acting musical dreams los angeles bittersweet heartbreak hollywood dancing", "poster": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=500&auto=format&fit=crop"},

    {"movie_id": 20, "title": "Whiplash", "genres": "Drama|Music", "director": "Damien Chazelle",
     "cast": "Miles Teller|J.K. Simmons|Paul Reiser|Melissa Benoist", "release_year": 2014, "vote_average": 8.5,
     "overview": "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student's potential.",
     "keywords": "jazz drumming perfectionism abuse obsession tempo intensity conservatory", "poster": "https://images.unsplash.com/photo-1519892300165-cb5542fb47c7?w=500&auto=format&fit=crop"},

    {"movie_id": 21, "title": "Her", "genres": "Drama|Romance|Sci-Fi", "director": "Spike Jonze",
     "cast": "Joaquin Phoenix|Scarlett Johansson|Amy Adams|Rooney Mara", "release_year": 2013, "vote_average": 8.0,
     "overview": "In a near future, a lonely writer develops an unlikely relationship with an operating system designed to meet his every need.",
     "keywords": "artificial intelligence lonely romance future operating system emotional connection intimacy", "poster": "https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=500&auto=format&fit=crop"},

    {"movie_id": 22, "title": "Eternal Sunshine of the Spotless Mind", "genres": "Drama|Romance|Sci-Fi", "director": "Michel Gondry",
     "cast": "Jim Carrey|Kate Winslet|Kirsten Dunst|Mark Ruffalo", "release_year": 2004, "vote_average": 8.3,
     "overview": "When their relationship turns sour, a couple undergoes a medical procedure to have each other erased from their memories.",
     "keywords": "memory erasure heartbreak surreal subconscious love nostalgia train montauk", "poster": "https://images.unsplash.com/photo-1518895949257-7621c3c786d7?w=500&auto=format&fit=crop"},

    {"movie_id": 23, "title": "The Notebook", "genres": "Drama|Romance", "director": "Nick Cassavetes",
     "cast": "Ryan Gosling|Rachel McAdams|James Garner|Gena Rowlands", "release_year": 2004, "vote_average": 7.8,
     "overview": "An elderly man reads to a woman with dementia the story of two young lovers whose romance is threatened by the difference in their respective social classes.",
     "keywords": "summer romance letters rain kiss social class memory alzheimers everlasting love", "poster": "https://images.unsplash.com/photo-1516589178581-6cd7833ae3b2?w=500&auto=format&fit=crop"},

    {"movie_id": 24, "title": "Crazy Stupid Love", "genres": "Comedy|Drama|Romance", "director": "Glenn Ficarra|John Requa",
     "cast": "Steve Carell|Ryan Gosling|Julianne Moore|Emma Stone", "release_year": 2011, "vote_average": 7.4,
     "overview": "A middle-aged husband's life changes dramatically when his wife asks him for a divorce. He seeks to rediscover his manhood with the help of a newfound friend, Jacob, learning to pick up girls at bars.",
     "keywords": "divorce makeover wingman romance dating twist family love hilarious", "poster": "https://images.unsplash.com/photo-1518199266791-5375a83190b7?w=500&auto=format&fit=crop"},

    # Animation / Pixar / Ghibli
    {"movie_id": 25, "title": "Spirited Away", "genres": "Animation|Adventure|Family|Fantasy|Mystery", "director": "Hayao Miyazaki",
     "cast": "Rumi Hiiragi|Miyu Irino|Mari Natsuki|Takashi Naitô", "release_year": 2001, "vote_average": 8.6,
     "overview": "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches and spirits, and where humans are changed into beasts.",
     "keywords": "bathhouse spirits yubaba dragon haku magic parents pigs japanese folklore", "poster": "https://images.unsplash.com/photo-1578632767115-351597cf2477?w=500&auto=format&fit=crop"},

    {"movie_id": 26, "title": "Princess Mononoke", "genres": "Animation|Action|Adventure|Fantasy", "director": "Hayao Miyazaki",
     "cast": "Yôji Matsuda|Yuriko Ishida|Yûko Tanaka|Kaoru Kobayashi", "release_year": 1997, "vote_average": 8.4,
     "overview": "On a journey to find the cure for a Tatarigami's curse, Ashitaka finds himself in the middle of a war between the forest gods and Tatara, a mining colony.",
     "keywords": "nature vs civilization forest spirit wolf iron town curse environmentalism gods", "poster": "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=500&auto=format&fit=crop"},

    {"movie_id": 27, "title": "Spider-Man: Into the Spider-Verse", "genres": "Animation|Action|Adventure|Sci-Fi", "director": "Bob Persichetti|Peter Ramsey",
     "cast": "Shameik Moore|Jake Johnson|Hailee Steinfeld|Mahershala Ali", "release_year": 2018, "vote_average": 8.4,
     "overview": "Teen Miles Morales becomes the new Spider-Man and joins other Spider-heroes from parallel dimensions to stop a threat to all reality.",
     "keywords": "multiverse leap of faith comic book style alternate dimensions spider-man brooklyn", "poster": "https://images.unsplash.com/photo-1635805737707-575885ab0820?w=500&auto=format&fit=crop"},

    {"movie_id": 28, "title": "WALL·E", "genres": "Animation|Adventure|Family|Sci-Fi", "director": "Andrew Stanton",
     "cast": "Ben Burtt|Elissa Knight|Jeff Garlin|Fred Willard", "release_year": 2008, "vote_average": 8.4,
     "overview": "In the distant future, a small waste-collecting robot inadvertently embarks on a space journey that will ultimately decide the fate of mankind.",
     "keywords": "robot waste earth desolate space love eve plant consumerism future humanity", "poster": "https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=500&auto=format&fit=crop"},

    # Mind-Bending / Mystery / Thriller
    {"movie_id": 29, "title": "Shutter Island", "genres": "Mystery|Thriller", "director": "Martin Scorsese",
     "cast": "Leonardo DiCaprio|Mark Ruffalo|Ben Kingsley|Michelle Williams", "release_year": 2010, "vote_average": 8.2,
     "overview": "In 1954, a U.S. Marshal investigates the disappearance of a murderer who escaped from a hospital for the criminally insane.",
     "keywords": "mental asylum island lighthouse hallucination twist trauma conspiracy psychological", "poster": "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=500&auto=format&fit=crop"},

    {"movie_id": 30, "title": "Parasite", "genres": "Comedy|Drama|Thriller", "director": "Bong Joon Ho",
     "cast": "Song Kang-ho|Lee Sun-kyun|Cho Yeo-jeong|Choi Woo-shik", "release_year": 2019, "vote_average": 8.5,
     "overview": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.",
     "keywords": "class conflict basement deception architecture rain social commentary south korea", "poster": "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop"}
]

def generate_ratings(movies_df, n_users=60, seed=42):
    """
    Generate realistic user ratings matrix reflecting distinct taste clusters:
    - Persona A (Users 1-15): Sci-Fi / Nolan / Mind-Bending Fans
    - Persona B (Users 16-30): Crime / Thriller / Tarantino / Scorsese Fans
    - Persona C (Users 31-45): Romance / Drama / Music Fans
    - Persona D (Users 46-60): Animation / Broad Cinema Fans
    """
    np.random.seed(seed)
    random.seed(seed)

    ratings_list = []
    movie_ids = movies_df["movie_id"].tolist()
    
    # Map genre/director profile to movie IDs
    scifi_ids = movies_df[movies_df["genres"].str.contains("Sci-Fi")]["movie_id"].tolist()
    crime_ids = movies_df[movies_df["genres"].str.contains("Crime")]["movie_id"].tolist()
    romance_ids = movies_df[movies_df["genres"].str.contains("Romance|Music")]["movie_id"].tolist()
    animation_ids = movies_df[movies_df["genres"].str.contains("Animation|Family")]["movie_id"].tolist()

    for user_id in range(1, n_users + 1):
        if user_id <= 15:
            # Sci-Fi / Nolan Lover
            fav_ids = scifi_ids
            other_ids = [m for m in movie_ids if m not in fav_ids]
        elif user_id <= 30:
            # Crime / Noir Lover
            fav_ids = crime_ids
            other_ids = [m for m in movie_ids if m not in fav_ids]
        elif user_id <= 45:
            # Romance / Musical Lover
            fav_ids = romance_ids
            other_ids = [m for m in movie_ids if m not in fav_ids]
        else:
            # Animation / Eclectic
            fav_ids = animation_ids
            other_ids = [m for m in movie_ids if m not in fav_ids]

        # Rate 8-12 favorites highly (4.0 - 5.0)
        n_fav = min(len(fav_ids), random.randint(6, 10))
        chosen_favs = random.sample(fav_ids, n_fav)
        for mid in chosen_favs:
            r = random.choice([4.0, 4.5, 5.0, 5.0, 4.5])
            ratings_list.append({"user_id": user_id, "movie_id": mid, "rating": r})

        # Rate 4-8 other movies moderately or randomly (1.5 - 4.0)
        n_other = random.randint(3, 7)
        chosen_others = random.sample(other_ids, n_other)
        for mid in chosen_others:
            r = random.choice([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
            ratings_list.append({"user_id": user_id, "movie_id": mid, "rating": r})

    return pd.DataFrame(ratings_list)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    movies_df = pd.DataFrame(MOVIES_DATA)
    movies_csv = os.path.join(data_dir, "movies.csv")
    movies_df.to_csv(movies_csv, index=False)

    ratings_df = generate_ratings(movies_df, n_users=60)
    ratings_csv = os.path.join(data_dir, "ratings.csv")
    ratings_df.to_csv(ratings_csv, index=False)

    print(f"Generated {len(movies_df)} movies and {len(ratings_df)} user ratings.")
    print(f"Movies saved to: {movies_csv}")
    print(f"Ratings saved to: {ratings_csv}")

if __name__ == "__main__":
    main()
