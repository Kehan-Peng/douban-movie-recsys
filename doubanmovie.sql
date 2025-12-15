CREATE TABLE user (
    email VARCHAR(255) PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE movies (
    id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(255) NOT NULL,
    directors VARCHAR(255),
    types VARCHAR(255),
    country VARCHAR(255),
    casts VARCHAR(255),
    rate FLOAT DEFAULT 0,
    comment_len INT DEFAULT 0,
    release_year INT,
    duration INT,
    summary TEXT,
    cover_url VARCHAR(255),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE comments (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_email VARCHAR(255),
    movie_id INT NOT NULL,
    content TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_comments_user FOREIGN KEY (user_email) REFERENCES user(email),
    CONSTRAINT fk_comments_movie FOREIGN KEY (movie_id) REFERENCES movies(id)
);

CREATE TABLE user_behavior (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_email VARCHAR(255) NOT NULL,
    movie_id INT NOT NULL,
    behavior_type TINYINT NOT NULL COMMENT '1-评分，2-收藏，3-观看',
    score FLOAT NULL COMMENT '仅评分行为有效，范围 0-10',
    create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_behavior_user FOREIGN KEY (user_email) REFERENCES user(email),
    CONSTRAINT fk_behavior_movie FOREIGN KEY (movie_id) REFERENCES movies(id),
    UNIQUE KEY uniq_user_movie_behavior (user_email, movie_id, behavior_type)
);

CREATE INDEX idx_movies_types ON movies(types);
CREATE INDEX idx_movies_directors ON movies(directors);
CREATE INDEX idx_movies_country ON movies(country);
CREATE INDEX idx_movies_casts ON movies(casts);
CREATE INDEX idx_behavior_user_email ON user_behavior(user_email);
CREATE INDEX idx_behavior_movie_id ON user_behavior(movie_id);
