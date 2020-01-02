# psuedo-code

def func(match):
    boundaries = cannyEdgeDetection(tennis court)
    data = trainCNN('tennis ball', 'players', 'obstacles')
    game = match.status()
    while ingame:
        photo = StereoPhoto()
        ball = data.findBalls()
        players = data.findPlayers()
        obstacles = data.findObstacles()
        
        if game.afterPlayerWonAPoint:
            while not robot.atTheBall():
                ballLocationRoute = findRoute(ball, players, obstacles)
                robot.moveToLocation(ballLocationRoute)

            if robot.atTheBall():
                robot.grabObject(ball)
            
            if robot.hasBall() and game.playerRequestABall():
                playerLocationRoute = findRoute(player, players, obstacles)
                robot.moveToLocation(playerLocationRoute)
                robot.faceDirection(player)
            
        robot.moveToLocation(defaultLocation)
        robot.victoryDanceAtLocation(defaultLocation)
    # game finished
    while court.hasBall():
        ballLocationRoute = findRoute(ball, players, obstacles)
        robot.moveToLocation(ballLocationRoute)
        if robot.atTheBall():
            robot.grabObject(ball)
    robot.moveToLocation(defaultLocation)
