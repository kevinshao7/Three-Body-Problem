#best estimate
r = []
v = []
#iterate positions
for body in 1:3
    for depth in 1:4 #search depth
        #search iteration
        results = zeros(Float128, (11, 11, 11))
        for i in -5:5
            for j in -5:5
                for k in -5:5
                    temp_r[body,:] = r[body,:] + [i*10^-(depth+1) j*10^-(depth+1) k*10^-(depth+1)]
                    error = eval 

                    results[i+6,j+6,k+6] = 

                end
            end
        end
        argmax(results)
    end
end