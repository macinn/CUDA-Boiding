#define FISH_H 0.5f
#define FISH_W 0.25f

#define FISH_NO_VERTICES 16
#define FISH_NO_INDICES 18

#define FISH_VERTICES \
    /* Triangle front */ \
    glm::vec3(0.f, 0.f + FISH_H / 2, 0.f),                              glm::vec3(0.f, 0.f, 1.f), \
    glm::vec3(0.f - FISH_W / 2, 0.f - FISH_H / 2, 0.f + FISH_W / 2),    glm::vec3(0.f, 0.f, 1.f), \
    glm::vec3(0.f + FISH_W / 2, 0.f - FISH_H / 2, 0.f + FISH_W / 2),    glm::vec3(0.f, 0.f, 1.f), \
    \
    /* Triangle left */ \
    glm::vec3(0.f, 0.f + FISH_H / 2, 0.f),                              glm::vec3(-1.f, 0.f, 0.f), \
    glm::vec3(0.f - FISH_W / 2, 0.f - FISH_H / 2, 0.f - FISH_W / 2),    glm::vec3(-1.f, 0.f, 0.f), \
    glm::vec3(0.f - FISH_W / 2, 0.f - FISH_H / 2, 0.f + FISH_W / 2),    glm::vec3(-1.f, 0.f, 0.f), \
    \
    /* Triangle back */ \
    glm::vec3(0.f, 0.f + FISH_H / 2, 0.f),                              glm::vec3(0.f, 0.f, -1.f), \
    glm::vec3(0.f + FISH_W / 2, 0.f - FISH_H / 2, 0.f - FISH_W / 2),    glm::vec3(0.f, 0.f, -1.f), \
    glm::vec3(0.f - FISH_W / 2, 0.f - FISH_H / 2, 0.f - FISH_W / 2),    glm::vec3(0.f, 0.f, -1.f), \
    \
    /* Triangles right */ \
    glm::vec3(0.f, 0.f + FISH_H / 2, 0.f),                              glm::vec3(1.f, 0.f, 0.f), \
    glm::vec3(0.f + FISH_W / 2, 0.f - FISH_H / 2, 0.f + FISH_W / 2),    glm::vec3(1.f, 0.f, 0.f), \
    glm::vec3(0.f + FISH_W / 2, 0.f - FISH_H / 2, 0.f - FISH_W / 2),    glm::vec3(1.f, 0.f, 0.f), \
    \
    /* Triangles bottom */ \
    glm::vec3(0.f + FISH_W / 2, 0.f - FISH_H / 2, 0.f + FISH_W / 2),    glm::vec3(0.f, -1.f, 0.f), \
    glm::vec3(0.f + FISH_W / 2, 0.f - FISH_H / 2, 0.f - FISH_W / 2),    glm::vec3(0.f, -1.f, 0.f), \
    glm::vec3(0.f - FISH_W / 2, 0.f - FISH_H / 2, 0.f + FISH_W / 2),    glm::vec3(0.f, -1.f, 0.f), \
    glm::vec3(0.f - FISH_W / 2, 0.f - FISH_H / 2, 0.f - FISH_W / 2),    glm::vec3(0.f, -1.f, 0.f)

#define FISH_INDICES \
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
