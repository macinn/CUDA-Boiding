// BOID MODEL SIZE
#define BOID_H 0.5f
#define BOID_W 0.25f

// BOID MODEL
#define BOID_NO_VERTICES 16
#define BOID_NO_INDICES 18

#define BOID_VERTICES \
    /* Triangle front */ \
    glm::vec3(0.f, 0.f + BOID_H / 2, 0.f),                              glm::vec3(0.f, 0.f, 1.f), \
    glm::vec3(0.f - BOID_W / 2, 0.f - BOID_H / 2, 0.f + BOID_W / 2),    glm::vec3(0.f, 0.f, 1.f), \
    glm::vec3(0.f + BOID_W / 2, 0.f - BOID_H / 2, 0.f + BOID_W / 2),    glm::vec3(0.f, 0.f, 1.f), \
    \
    /* Triangle left */ \
    glm::vec3(0.f, 0.f + BOID_H / 2, 0.f),                              glm::vec3(-1.f, 0.f, 0.f), \
    glm::vec3(0.f - BOID_W / 2, 0.f - BOID_H / 2, 0.f - BOID_W / 2),    glm::vec3(-1.f, 0.f, 0.f), \
    glm::vec3(0.f - BOID_W / 2, 0.f - BOID_H / 2, 0.f + BOID_W / 2),    glm::vec3(-1.f, 0.f, 0.f), \
    \
    /* Triangle back */ \
    glm::vec3(0.f, 0.f + BOID_H / 2, 0.f),                              glm::vec3(0.f, 0.f, -1.f), \
    glm::vec3(0.f + BOID_W / 2, 0.f - BOID_H / 2, 0.f - BOID_W / 2),    glm::vec3(0.f, 0.f, -1.f), \
    glm::vec3(0.f - BOID_W / 2, 0.f - BOID_H / 2, 0.f - BOID_W / 2),    glm::vec3(0.f, 0.f, -1.f), \
    \
    /* Triangles right */ \
    glm::vec3(0.f, 0.f + BOID_H / 2, 0.f),                              glm::vec3(1.f, 0.f, 0.f), \
    glm::vec3(0.f + BOID_W / 2, 0.f - BOID_H / 2, 0.f + BOID_W / 2),    glm::vec3(1.f, 0.f, 0.f), \
    glm::vec3(0.f + BOID_W / 2, 0.f - BOID_H / 2, 0.f - BOID_W / 2),    glm::vec3(1.f, 0.f, 0.f), \
    \
    /* Triangles bottom */ \
    glm::vec3(0.f + BOID_W / 2, 0.f - BOID_H / 2, 0.f + BOID_W / 2),    glm::vec3(0.f, -1.f, 0.f), \
    glm::vec3(0.f + BOID_W / 2, 0.f - BOID_H / 2, 0.f - BOID_W / 2),    glm::vec3(0.f, -1.f, 0.f), \
    glm::vec3(0.f - BOID_W / 2, 0.f - BOID_H / 2, 0.f + BOID_W / 2),    glm::vec3(0.f, -1.f, 0.f), \
    glm::vec3(0.f - BOID_W / 2, 0.f - BOID_H / 2, 0.f - BOID_W / 2),    glm::vec3(0.f, -1.f, 0.f)

#define BOID_INDICES \
    /* Triangle front */ \
    0, 1, 2, \
    /* Triangle left */ \
    3, 4, 5, \
    /* Triangle back */ \
    6, 7, 8, \
    /* Triangle right */ \
    9, 10, 11, \
    /* Triangles bottom */ \
    12, 14, 13, \
    13, 14, 15
