set(BUILD_NUMBER 0)
set(BUILD_COMMIT "unknown")
set(BUILD_COMPILER "unknown")
set(BUILD_TARGET "unknown")

# GreenBoost: this tree is vendored into the greenboost repo with no .git of
# its own (see PINNED_COMMIT next to this file). Without the EXISTS guard
# below, `git rev-parse`/`git rev-list` walk up from
# CMAKE_CURRENT_SOURCE_DIR and silently succeed against greenboost's own
# enclosing .git, baking GREENBOOST's commit hash/count into
# llama-server --version, /props's build_info, and every OpenAI-compat
# response's system_fingerprint as if it were llama.cpp's own build
# identity. Guard the same way common/CMakeLists.txt already guards its own
# CONFIGURE_DEPENDS use of .git (line ~11), and fall back to the tree's own
# PINNED_COMMIT (the same file GreenBoost's engine.version is read from) so
# llama.cpp's self-reported identity agrees with GreenBoost's instead of
# reporting the wrong repo's history.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    # Look for git
    find_package(Git)
    if(NOT Git_FOUND)
        find_program(GIT_EXECUTABLE NAMES git git.exe)
        if(GIT_EXECUTABLE)
            set(Git_FOUND TRUE)
            message(STATUS "Found Git: ${GIT_EXECUTABLE}")
        else()
            message(WARNING "Git not found. Build info will not be accurate.")
        endif()
    endif()

    # Get the commit count and hash
    if(Git_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE HEAD
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RES
        )
        if (RES EQUAL 0)
            set(BUILD_COMMIT ${HEAD})
        endif()
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-list --count HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE COUNT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RES
        )
        if (RES EQUAL 0)
            set(BUILD_NUMBER ${COUNT})
        endif()
    endif()
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/PINNED_COMMIT")
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/PINNED_COMMIT" PINNED_FULL LIMIT_COUNT 1)
    string(SUBSTRING "${PINNED_FULL}" 0 9 BUILD_COMMIT)
endif()

set(BUILD_COMPILER "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

if(CMAKE_VS_PLATFORM_NAME)
    set(BUILD_TARGET ${CMAKE_VS_PLATFORM_NAME})
else()
    set(BUILD_TARGET "${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
endif()
