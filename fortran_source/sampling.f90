module sampling
    use MTJ_RNG_vars
    use ziggurat
    use joule_heating
    implicit none
    logical  :: fwrite_enabled
    real(dp) :: T_free, T_bath
    contains
        ! --------------------------------*------------*-----------------------------------
        !   energy_usage,bit,theta_end,phi_end, should not be passed into this function
        !   they are declared as they are because subroutine arguments declared with intent(out)
        !   will return back to python as a tuple (see numpy f2py)
        !
        !   intended to mimic the python function call: out,energy = dev.sample_*(Jappl, [device params]....)
        ! --------------------------------*------------*-----------------------------------
        subroutine sample_SHE(energy_usage, bit, theta_end, phi_end,&
                                 Jappl, Jshe, Hy_in, theta_init, phi_init, Ki_in, TMR_in, Rp_in,&
                                 a_in, b_in, tf_in, alpha_in, Ms_in, eta_in, d_in, tox_in, t_pulse, t_relax,&
                                 T_in, Nx_in, Ny_in, Nz_in,&
                                 dump_mod, view_mag_flag, sample_count, file_ID, heating_enabled)
            implicit none
            integer, parameter :: dp = kind(0.0d0)
            ! Dynamical parameters
            real, intent(in) :: Jappl, Jshe, theta_init, phi_init, t_pulse, t_relax, T_in
            ! Device input parameters
            real, intent(in) :: Ki_in, TMR_in, Rp_in, Ms_in, Hy_in,&
                                a_in, b_in, d_in, tf_in, alpha_in,&
                                eta_in, tox_in, Nx_in, Ny_in, Nz_in
            ! Functional parameters
            integer, intent(in) :: file_ID, sample_count, dump_mod
            logical, intent(in) :: view_mag_flag, heating_enabled
            ! Return values
            real, intent(out) :: energy_usage, theta_end, phi_end
            integer, intent(out) :: bit
            !==================================================================
            real(dp), dimension(:), allocatable :: theta_evol, phi_evol, temp_evol
            real(dp) :: phi_i, theta_i, cuml_pow
            real :: seed
            integer :: t_i, pulse_steps, relax_steps, total_steps
            !==================================================================
            !//////////////////////////////////////////////////////////////////

            ! ======== solve init =========
            ! Fortran array indexing starts at 1
            t_i  = 1
            fwrite_enabled = ((mod(sample_count,dump_mod) .eq. 0 .and. view_mag_flag))

            pulse_steps = int(t_pulse/t_step)
            relax_steps = int(t_relax/t_step)
            total_steps = pulse_steps+relax_steps+1

            cuml_pow = 0.0_dp
            theta_i = real(theta_init, dp)
            phi_i   = real(phi_init, dp)
            T_free = real(T_in, dp)
            T_bath = real(T_in, dp)
            T      = real(T_in, dp)
            if(fwrite_enabled) then
                allocate(theta_evol(total_steps))
                allocate(phi_evol(total_steps))
                allocate(temp_evol(total_steps))
                theta_evol(t_i) = theta_i
                phi_evol(t_i)   = phi_i
                temp_evol(t_i) = T
            end if

            Ki = real(Ki_in, dp)
            Ms = real(Ms_in, dp)
            call set_params(TMR_in, Rp_in, alpha_in, tf_in, a_in, b_in, d_in,&
                            eta_in, tox_in, Nx_in, Ny_in, Nz_in)

            call random_number(seed)
            call zigset(int(1+floor((1000001)*seed)))
            !================================

            Hy = real(Hy_in,dp)
            !=========== Pulse current and set device to be in-plane =========
            call drive(0.0_dp, real(Jshe,dp), real(Jappl,dp), 0.0_dp, 0.0_dp, pulse_steps,&
                           t_i, phi_i, theta_i, phi_evol, theta_evol, temp_evol, cuml_pow,&
                           heating_enabled)

            Hy = 0
            !=================  Relax into one of two low-energy states out-of-plane  ================
            call drive(0.0_dp, 0.0_dp, real(Jappl,dp), 0.0_dp, 0.0_dp, relax_steps,&
                           t_i, phi_i, theta_i, phi_evol, theta_evol, temp_evol, cuml_pow,&
                           heating_enabled)

            if(fwrite_enabled) then
                call file_dump(file_ID, phi_evol, theta_evol, temp_evol, 1)
            end if

            ! ===== return final solve values: energy,bit,theta,phi ====
            if( cos(theta_i) > 0.0_dp ) then
                bit = 1
            else
                bit = 0
            end if
            ! Cast to real before returning to Python
            theta_end = real(theta_i)
            phi_end   = real(phi_i)
            energy_usage = real(cuml_pow*t_step)
        end subroutine sample_SHE

        subroutine sample_SWrite(energy_usage, bit, theta_end, phi_end,&
                                 Jappl, Jreset, Hreset, theta_init, phi_init, K_295_in, TMR_in, Rp_in,&
                                 a_in, b_in, tf_in, alpha_in, Ms_295_in, eta_in, d_in, tox_in, t_pulse,&
                                 t_relax, t_reset, T_in, Nx_in, Ny_in, Nz_in,&
                                 dump_mod, view_mag_flag, sample_count, file_ID, heating_enabled)
            implicit none
            integer, parameter :: dp = kind(0.0d0)
            ! Dynamical parameters
            real, intent(in) :: Jappl, Jreset, Hreset, theta_init, phi_init,&
                                t_pulse, t_relax, t_reset, T_in
            ! Device input parameters
            real, intent(in) :: K_295_in, TMR_in, Rp_in, Ms_295_in,&
                                a_in, b_in, d_in, tf_in, alpha_in,&
                                eta_in, tox_in, Nx_in, Ny_in, Nz_in
            ! Functional parameters
            integer, intent(in) :: file_ID, sample_count, dump_mod
            logical, intent(in) :: view_mag_flag, heating_enabled
            ! Return values
            real, intent(out) :: energy_usage, theta_end, phi_end
            integer, intent(out) :: bit
            !==================================================================
            real(dp), dimension(:), allocatable :: theta_evol, phi_evol, temp_evol
            real(dp) :: phi_i, theta_i, cuml_pow, K_295, Ms_295
            real :: seed
            integer :: t_i, pulse_steps, relax_steps, reset_steps,&
                       sample_steps, total_reset_steps
            !==================================================================
            !//////////////////////////////////////////////////////////////////

            ! ======== solve init =========
            ! Fortran array indexing starts at 1
            t_i  = 1
            fwrite_enabled = ((mod(sample_count,dump_mod) .eq. 0 .and. view_mag_flag))

            pulse_steps = int(t_pulse/t_step)
            relax_steps = int(t_relax/t_step)
            reset_steps = int(t_reset/t_step)
            sample_steps = pulse_steps+relax_steps+1
            total_reset_steps = reset_steps+relax_steps

            cuml_pow = 0.0_dp
            theta_i = real(theta_init, dp)
            phi_i   = real(phi_init, dp)
            K_295  = real(K_295_in, dp)
            Ms_295 = real(Ms_295_in, dp)
            T_free = real(T_in, dp)
            T_bath = real(T_in, dp)
            T      = real(T_in, dp)
            if(fwrite_enabled) then
                allocate(theta_evol(sample_steps))
                allocate(phi_evol(sample_steps))
                allocate(temp_evol(sample_steps))
                theta_evol(t_i) = theta_i
                phi_evol(t_i)   = phi_i
                temp_evol(t_i) = T_free
            end if

            ! compute K and Ms with temperature dependence
            !sets K, Ms
            call compute_K_and_Ms(K_295, Ms_295, T_free)
            !redundant, sets K, Ms, and Bsat with no change. Avoids conflict with other device models
            call set_params(TMR_in, Rp_in, alpha_in, tf_in, a_in, b_in, d_in,&
                            eta_in, tox_in, Nx_in, Ny_in, Nz_in)

            call random_number(seed)
            call zigset(int(1+floor((1000001)*seed)))
            !================================

            call set_layers(A1)

            Hz = 0.0_dp
            call drive(0.0_dp, 0.0_dp, real(Jappl,dp), K_295, Ms_295, pulse_steps,&
                           t_i, phi_i, theta_i, phi_evol, theta_evol, temp_evol, cuml_pow,&
                           heating_enabled)

            call drive(0.0_dp, 0.0_dp, 0.0_dp, K_295, Ms_295, relax_steps,&
                           t_i, phi_i, theta_i, phi_evol, theta_evol, temp_evol, cuml_pow,&
                           heating_enabled)

            if( cos(theta_i) > 0.0_dp ) then
                bit = 1
            else
                bit = 0
            end if

            ! avoiding stack overflow by having two write operations since f2py forcefully copies arrays
            if(fwrite_enabled) then
                call file_dump(file_ID, phi_evol, theta_evol, temp_evol, 1)
                deallocate(phi_evol)
                deallocate(theta_evol)
                deallocate(temp_evol)
                t_i = 0
                allocate(phi_evol(total_reset_steps))
                allocate(theta_evol(total_reset_steps))
                allocate(temp_evol(total_reset_steps))
            end if

            Hz = Hreset
            call drive(0.0_dp, 0.0_dp, real(Jreset,dp), K_295, Ms_295, reset_steps,&
                           t_i, phi_i, theta_i, phi_evol, theta_evol, temp_evol, cuml_pow,&
                           heating_enabled)

            Hz = 0.0_dp
            call drive(0.0_dp, 0.0_dp, 0.0_dp, K_295, Ms_295, relax_steps,&
                           t_i, phi_i, theta_i, phi_evol, theta_evol, temp_evol, cuml_pow,&
                           heating_enabled)

            if(fwrite_enabled) then
                call file_dump(file_ID, phi_evol, theta_evol, temp_evol, 0)
                deallocate(phi_evol)
                deallocate(theta_evol)
                deallocate(temp_evol)
            end if

            ! ===== return final solve values: energy,bit,theta,phi ====
            ! Cast to real before returning to Python
            theta_end = real(theta_i)
            phi_end   = real(phi_i)
            energy_usage = real(cuml_pow*t_step)
        end subroutine sample_SWrite

        subroutine sample_VCMA(energy_usage, bit, theta_end, phi_end,&
                                 Jappl, v_pulse, theta_init, phi_init, Ki_in, TMR_in, Rp_in,&
                                 a_in, b_in, tf_in, alpha_in, Ms_in, eta_in, d_in, tox_in,&
                                 t_pulse, t_relax, T_in, Nx_in, Ny_in, Nz_in,&
                                 dump_mod, view_mag_flag, sample_count, file_ID, heating_enabled)
            implicit none
            integer, parameter :: dp = kind(0.0d0)
            ! Dynamical parameters
            real, intent(in) :: Jappl, v_pulse, theta_init, phi_init, t_pulse, t_relax, T_in
            ! Device input parameters
            real, intent(in) :: Ki_in, TMR_in, Rp_in, Ms_in,&
                                a_in, b_in, d_in, tf_in, alpha_in, eta_in, tox_in,&
                                Nx_in, Ny_in, Nz_in
            ! Functional parameters
            integer, intent(in) :: file_ID, sample_count, dump_mod
            logical, intent(in) :: view_mag_flag, heating_enabled
            ! Return values
            real, intent(out) :: energy_usage, theta_end, phi_end
            integer, intent(out) :: bit
            !==================================================================
            real(dp), dimension(:), allocatable :: theta_evol, phi_evol, temp_evol
            real(dp) :: phi_i, theta_i, cuml_pow
            real :: seed
            integer :: t_i, pulse_steps, relax_steps, total_steps
            !==================================================================
            !//////////////////////////////////////////////////////////////////

            ! ======== solve init =========
            ! Fortran array indexing starts at 1
            t_i  = 1
            fwrite_enabled = ((mod(sample_count,dump_mod) .eq. 0 .and. view_mag_flag))

            pulse_steps = int(t_pulse/t_step)
            relax_steps = int(t_relax/t_step)
            total_steps = pulse_steps+relax_steps+1

            cuml_pow = 0.0_dp
            theta_i = real(theta_init, dp)
            phi_i   = real(phi_init, dp)
            T_free = real(T_in, dp)
            T_bath = real(T_in, dp)
            T      = real(T_in, dp)
            if(fwrite_enabled) then
                allocate(theta_evol(total_steps))
                allocate(phi_evol(total_steps))
                allocate(temp_evol(total_steps))
                theta_evol(t_i) = theta_i
                phi_evol(t_i)   = phi_i
                temp_evol(t_i)  = T
            end if

            Ki = real(Ki_in, dp)
            Ms = real(Ms_in, dp)
            call set_params(TMR_in, Rp_in, alpha_in, tf_in, a_in, b_in, d_in,&
                            eta_in, tox_in, Nx_in, Ny_in, Nz_in)

            call random_number(seed)
            call zigset(int(1+floor((1000001)*seed)))
            !================================

            call drive(real(v_pulse,dp), 0.0_dp, real(Jappl,dp),0.0_dp , 0.0_dp, pulse_steps,&
                           t_i, phi_i, theta_i, phi_evol, theta_evol, temp_evol, cuml_pow, heating_enabled)

            call drive(0.0_dp, 0.0_dp, real(Jappl,dp), 0.0_dp, 0.0_dp, relax_steps,&
                           t_i, phi_i, theta_i, phi_evol, theta_evol, temp_evol, cuml_pow, heating_enabled)

            if(fwrite_enabled) then
                call file_dump(file_ID, phi_evol, theta_evol, temp_evol, 1)
            end if

            ! ===== return final solve values: energy,bit,theta,phi ====
            if( cos(theta_i) > 0.0_dp ) then
                bit = 1
            else
                bit = 0
            end if
            ! Cast to real before returning to Python
            theta_end = real(theta_i)
            phi_end   = real(phi_i)
            energy_usage = real(cuml_pow*t_step)
          end subroutine sample_VCMA

          subroutine check_SHE(mz_c1, mz_c2, p2pv,&
               Jappl, Jshe, Hy_in, theta_init, phi_init, Ki_in, TMR_in, Rp_in,&
               a_in, b_in, tf_in, alpha_in, Ms_in, eta_in, d_in, tox_in, t_pulse, t_relax,&
               T_in, Nx_in, Ny_in, Nz_in, heating_enabled, cyc, pcs, rcs)

            !++++ Configuration check for SHE device ++++++
            ! this was formerly done via output files and interacting with python. in order to avoid
            ! the need to write files to interact with python, the entire functionality was moved here

            implicit none
            integer, parameter :: dp = kind(0.0d0)
            ! Dynamical parameters
            real, intent(in) :: Jappl, Jshe, theta_init, phi_init, t_pulse, t_relax, T_in
            ! Device input parameters
            real, intent(in) :: Ki_in, TMR_in, Rp_in, Ms_in, Hy_in,&
                                a_in, b_in, d_in, tf_in, alpha_in,&
                                eta_in, tox_in, Nx_in, Ny_in, Nz_in
            ! Functional parameters
            integer, intent(in) :: cyc, pcs, rcs
            logical, intent(in) :: heating_enabled
            ! Return values
            real, intent(out) :: mz_c1, mz_c2, p2pv
            !==================================================================
            real(dp), dimension(:), allocatable :: theta_evol, phi_evol, temp_evol,&
                                                   mz_c1_arr, mz_c2_arr, mz_arr
            real(dp) :: phi_i, theta_i, cuml_pow
            real :: seed
            integer :: t_i, pulse_steps, relax_steps, total_steps, c_i, m_i
            !==================================================================
            !//////////////////////////////////////////////////////////////////

            ! ======== solve init =========
            ! Fortran array indexing starts at 1

            fwrite_enabled = .true.

            p2pv = 0.0; mz_c1 = 0.0; mz_c2 = 0.0

            pulse_steps = int(t_pulse/t_step)
            relax_steps = int(t_relax/t_step)

            total_steps = pulse_steps+relax_steps+1

            allocate(mz_c1_arr(pulse_steps))
            allocate(mz_c2_arr(relax_steps))
            allocate(mz_arr(total_steps))

            mz_c1_arr = 0.0; mz_c2_arr = 0.0

            T_free = real(T_in, dp)
            T_bath = real(T_in, dp)
            T      = real(T_in, dp)

            allocate(theta_evol(total_steps))
            allocate(phi_evol(total_steps))
            allocate(temp_evol(total_steps))

            Ki = real(Ki_in, dp)
            Ms = real(Ms_in, dp)
            call set_params(TMR_in, Rp_in, alpha_in, tf_in, a_in, b_in, d_in,&
                            eta_in, tox_in, Nx_in, Ny_in, Nz_in)

            call random_number(seed)
            call zigset(int(1+floor((1000001)*seed)))
            !================================

            Hy = real(Hy_in,dp)

            do c_i = 1,cyc
               t_i  = 1
               cuml_pow = 0.0_dp
               theta_i = real(theta_init, dp)
               phi_i   = real(phi_init, dp)
               theta_evol(t_i) = theta_i
               phi_evol(t_i)   = phi_i
               temp_evol(t_i)  = T
               mz_arr = 0.0
               !=========== Pulse current and set device to be in-plane =========
               call drive(0.0_dp, real(Jshe,dp), real(Jappl,dp), 0.0_dp, 0.0_dp, pulse_steps,&
                    t_i, phi_i, theta_i, phi_evol, theta_evol, temp_evol, cuml_pow,&
                    heating_enabled)

               Hy = 0
               !=================  Relax into one of two low-energy states out-of-plane  ================
               call drive(0.0_dp, 0.0_dp, real(Jappl,dp), 0.0_dp, 0.0_dp, relax_steps,&
                    t_i, phi_i, theta_i, phi_evol, theta_evol, temp_evol, cuml_pow,&
                    heating_enabled)

               do m_i = 1,total_steps
                  mz_arr(m_i) = abs(cos(theta_evol(m_i)))
               end do

               mz_c1_arr = mz_c1_arr + mz_arr(1:pulse_steps)
               mz_c2_arr = mz_c2_arr + mz_arr(pulse_steps+2:total_steps)
               p2pv = p2pv + real(sum(abs(mz_arr(2:total_steps) &
                           - mz_arr(1:total_steps-1)))/real(total_steps-1))

            end do

            p2pv = p2pv/real(cyc)
            mz_c1 = real(sum(mz_c1_arr(pcs:pulse_steps)))/real(cyc)/real(pulse_steps-pcs)
            mz_c2 = real(sum(mz_c2_arr(rcs:relax_steps)))/(real(cyc)*real(relax_steps-rcs))

            deallocate(theta_evol,phi_evol,temp_evol,mz_c1_arr,mz_c2_arr,mz_arr)

          end subroutine check_SHE

          subroutine check_SWrite(mz_c1, mz_c2, p2pv,&
               Jappl, Jreset, Hreset, theta_init, phi_init, K_295_in, TMR_in, Rp_in,&
               a_in, b_in, tf_in, alpha_in, Ms_295_in, eta_in, d_in, tox_in, t_pulse,&
               t_relax, t_reset, T_in, Nx_in, Ny_in, Nz_in,&
               heating_enabled, cyc, pcs, rcs)
            implicit none
            integer, parameter :: dp = kind(0.0d0)
            ! Dynamical parameters
            real, intent(in) :: Jappl, Jreset, Hreset, theta_init, phi_init,&
                                t_pulse, t_relax, t_reset, T_in
            ! Device input parameters
            real, intent(in) :: K_295_in, TMR_in, Rp_in, Ms_295_in,&
                                a_in, b_in, d_in, tf_in, alpha_in,&
                                eta_in, tox_in, Nx_in, Ny_in, Nz_in
            ! Functional parameters
            integer, intent(in) :: cyc, pcs, rcs
            logical, intent(in) :: heating_enabled
            ! Return values
            real, intent(out) :: mz_c1, mz_c2, p2pv
            !==================================================================
            real(dp), dimension(:), allocatable :: theta_evol, phi_evol, temp_evol,&
                                                   mz_c1_arr, mz_c2_arr, mz_arr
            real(dp) :: phi_i, theta_i, cuml_pow, K_295, Ms_295
            real :: seed
            integer :: t_i, pulse_steps, relax_steps, reset_steps,&
                       sample_steps, total_reset_steps, total_steps, c_i, m_i
            !==================================================================
            !//////////////////////////////////////////////////////////////////

            ! ======== solve init =========
            ! Fortran array indexing starts at 1
            fwrite_enabled = .true.
            p2pv = 0.0; mz_c1 = 0.0; mz_c2 = 0.0

            pulse_steps = int(t_pulse/t_step)
            relax_steps = int(t_relax/t_step)
            reset_steps = int(t_reset/t_step)
            sample_steps = pulse_steps+relax_steps+1
            total_reset_steps = reset_steps+relax_steps
            total_steps = sample_steps + total_reset_steps

            allocate(mz_c1_arr(sample_steps))
            allocate(mz_c2_arr(total_reset_steps))
            allocate(mz_arr(total_steps))

            mz_c1_arr = 0.0; mz_c2_arr = 0.0

            K_295  = real(K_295_in, dp)
            Ms_295 = real(Ms_295_in, dp)

            call random_number(seed)
            call zigset(int(1+floor((1000001)*seed)))
            !================================

            call set_layers(A1)

            do c_i = 1,cyc
               t_i = 1
               cuml_pow = 0.0_dp
               theta_i = real(theta_init, dp)
               phi_i   = real(phi_init, dp)
               T_free = real(T_in, dp)
               T_bath = real(T_in, dp)
               T      = real(T_in, dp)
               call compute_K_and_Ms(K_295, Ms_295, T_free)
               call set_params(TMR_in, Rp_in, alpha_in, tf_in, a_in, b_in, d_in,&
                               eta_in, tox_in, Nx_in, Ny_in, Nz_in)

               allocate(theta_evol(sample_steps))
               allocate(phi_evol(sample_steps))
               allocate(temp_evol(sample_steps))
               theta_evol(t_i) = theta_i
               phi_evol(t_i)   = phi_i
               temp_evol(t_i)  = T_free
               mz_arr = 0.0

               Hz = 0.0_dp
               call drive(0.0_dp, 0.0_dp, real(Jappl,dp), K_295, Ms_295, pulse_steps,&
                    t_i, phi_i, theta_i, phi_evol, theta_evol, temp_evol, cuml_pow,&
                    heating_enabled)

               call drive(0.0_dp, 0.0_dp, 0.0_dp, K_295, Ms_295, relax_steps,&
                    t_i, phi_i, theta_i, phi_evol, theta_evol, temp_evol, cuml_pow,&
                    heating_enabled)

               do m_i = 1,sample_steps
                  mz_arr(m_i) = abs(cos(theta_evol(m_i)))
               end do

               ! avoiding stack overflow by having two write operations since f2py forcefully copies arrays
               deallocate(phi_evol)
               deallocate(theta_evol)
               deallocate(temp_evol)
               t_i = 0
               allocate(phi_evol(total_reset_steps))
               allocate(theta_evol(total_reset_steps))
               allocate(temp_evol(total_reset_steps))

               Hz = Hreset
               call drive(0.0_dp, 0.0_dp, real(Jreset,dp), K_295, Ms_295, reset_steps,&
                    t_i, phi_i, theta_i, phi_evol, theta_evol, temp_evol, cuml_pow,&
                    heating_enabled)

               Hz = 0.0_dp
               call drive(0.0_dp, 0.0_dp, 0.0_dp, K_295, Ms_295, relax_steps,&
                    t_i, phi_i, theta_i, phi_evol, theta_evol, temp_evol, cuml_pow,&
                    heating_enabled)

               do m_i = sample_steps+1,total_steps
                  mz_arr(m_i) = abs(cos(theta_evol(m_i-sample_steps)))
               end do

               mz_c1_arr = mz_c1_arr + mz_arr(1:sample_steps)
               mz_c2_arr = mz_c2_arr + mz_arr(sample_steps+1:total_steps)
               p2pv = p2pv + real(sum(abs(mz_arr(2:total_steps) &
                           - mz_arr(1:total_steps-1)))/real(total_steps-1))

               deallocate(phi_evol)
               deallocate(theta_evol)
               deallocate(temp_evol)
            end do

            ! ===== return final solve values: energy,bit,theta,phi ====
            p2pv = p2pv/real(cyc)
            mz_c1 = real(sum(mz_c1_arr(pcs:sample_steps)))/real(cyc)/real(sample_steps-pcs)
            mz_c2 = real(sum(mz_c2_arr(rcs:total_reset_steps)))/(real(cyc)*real(total_reset_steps-rcs))

            deallocate(mz_c1_arr,mz_c2_arr,mz_arr)

            return

          end subroutine check_SWrite

          subroutine check_VCMA(mz_c1, mz_c2, p2pv,&
               Jappl, v_pulse, theta_init, phi_init, Ki_in, TMR_in, Rp_in,&
               a_in, b_in, tf_in, alpha_in, Ms_in, eta_in, d_in, tox_in,&
               t_pulse, t_relax, T_in, Nx_in, Ny_in, Nz_in,&
               heating_enabled, cyc, pcs, rcs)

            !++++ Configuration check for VCMA device ++++++
            ! this was formerly done via output files and interacting with python. in order to avoid
            ! the need to write files to interact with python, the entire functionality was moved here

            implicit none
            integer, parameter :: dp = kind(0.0d0)
            ! Dynamical parameters
            real, intent(in) :: Jappl, v_pulse, theta_init, phi_init, t_pulse, t_relax, T_in
            ! Device input parameters
            real, intent(in) :: Ki_in, TMR_in, Rp_in, Ms_in,&
                                a_in, b_in, d_in, tf_in, alpha_in,&
                                eta_in, tox_in, Nx_in, Ny_in, Nz_in
            ! Functional parameters
            integer, intent(in) :: cyc, pcs, rcs
            logical, intent(in) :: heating_enabled
            ! Return values
            real, intent(out) :: mz_c1, mz_c2, p2pv
            !==================================================================
            real(dp), dimension(:), allocatable :: theta_evol, phi_evol, temp_evol,&
                                                   mz_c1_arr, mz_c2_arr, mz_arr
            real(dp) :: phi_i, theta_i, cuml_pow
            real :: seed
            integer :: t_i, pulse_steps, relax_steps, total_steps, c_i, m_i
            !==================================================================
            !//////////////////////////////////////////////////////////////////

            ! ======== solve init =========
            ! Fortran array indexing starts at 1

            fwrite_enabled = .true.

            Hy = 0

            p2pv = 0.0; mz_c1 = 0.0; mz_c2 = 0.0

            pulse_steps = int(t_pulse/t_step)
            relax_steps = int(t_relax/t_step)

            total_steps = pulse_steps+relax_steps+1

            allocate(mz_c1_arr(pulse_steps))
            allocate(mz_c2_arr(relax_steps))
            allocate(mz_arr(total_steps))

            mz_c1_arr = 0.0; mz_c2_arr = 0.0

            cuml_pow = 0.0_dp
            T_free = real(T_in, dp)
            T_bath = real(T_in, dp)
            T      = real(T_in, dp)

            allocate(theta_evol(total_steps))
            allocate(phi_evol(total_steps))
            allocate(temp_evol(total_steps))

            Ki = real(Ki_in, dp)
            Ms = real(Ms_in, dp)
            call set_params(TMR_in, Rp_in, alpha_in, tf_in, a_in, b_in, d_in,&
                            eta_in, tox_in, Nx_in, Ny_in, Nz_in)

            call random_number(seed)
            call zigset(int(1+floor((1000001)*seed)))
            !================================

            do c_i = 1,cyc
               t_i  = 1
               theta_i = real(theta_init, dp)
               phi_i   = real(phi_init, dp)
               theta_evol(t_i) = theta_i
               phi_evol(t_i)   = phi_i
               temp_evol(t_i) = T
               mz_arr = 0.0
               !=========== Pulse current and set device to be in-plane =========
               call drive(real(v_pulse,dp), 0.0_dp, real(Jappl,dp), 0.0_dp, 0.0_dp, pulse_steps,&
                    t_i, phi_i, theta_i, phi_evol, theta_evol, temp_evol, cuml_pow,&
                    heating_enabled)


               !=================  Relax into one of two low-energy states out-of-plane  ===================
               call drive(0.0_dp, 0.0_dp, real(Jappl,dp), 0.0_dp, 0.0_dp, relax_steps,&
                    t_i, phi_i, theta_i, phi_evol, theta_evol, temp_evol, cuml_pow,&
                    heating_enabled)

               do m_i = 1,total_steps
                  mz_arr(m_i) = abs(cos(theta_evol(m_i)))
               end do


               mz_c1_arr = mz_c1_arr + mz_arr(1:pulse_steps)
               mz_c2_arr = mz_c2_arr + mz_arr(pulse_steps+2:total_steps)
               p2pv = p2pv + real(sum(abs(mz_arr(2:total_steps) - mz_arr(1:total_steps-1)))/real(total_steps-1))

            end do

            p2pv = p2pv/real(cyc)
            mz_c1 = real(sum(mz_c1_arr(pcs:pulse_steps)))/real(cyc)/real(pulse_steps-pcs)
            mz_c2 = real(sum(mz_c2_arr(rcs:relax_steps)))/(real(cyc)*real(relax_steps-rcs))


            deallocate(theta_evol,phi_evol,temp_evol,mz_c1_arr,mz_c2_arr,mz_arr)
            
            return

          end subroutine check_VCMA

        subroutine drive(V, J_SHE, J_STT, K_295, Ms_295, steps, t_i, phi_i, theta_i,&
                         phi_evol, theta_evol, temp_evol, cuml_pow, heating_enabled)
           implicit none
           integer,  parameter :: dp = kind(0.0d0)
           real(dp), intent(in) :: V, J_SHE, J_STT, K_295, Ms_295
           integer,  intent(in)  :: steps
           logical,  intent(in)  :: heating_enabled
           real(dp), dimension(:), intent(inout) :: phi_evol, theta_evol, temp_evol
           real(dp), intent(inout) :: phi_i, theta_i, cuml_pow
           integer,  intent(inout) :: t_i
           real(dp) :: Hk, Ax, Ay, Az, dphi, dtheta, R1, pow, cos_theta, sin_theta,&
                       cos_phi, sin_phi, v_pow, she_pow, T_init
           integer  :: i

           Hk = ((2.0_dp*Ki)/(tf*Ms*u0)) - ((2.0_dp*ksi*V)/(u0*Ms*tox*tf))
           v_pow = 0.5_dp*cap_mgo*V**2
           she_pow = R2*(J_SHE*A2)**2
           T_init = T_free

           do i = 1, steps
               t_i = t_i+1
               cos_theta = cos(theta_i)
               sin_theta = sin(theta_i)
               cos_phi   = cos(phi_i)
               sin_phi   = sin(phi_i)
               Ax = Hx-Nx*Ms*sin_theta*cos_phi     +rnor()*Htherm
               Ay = Hy-Ny*Ms*sin_theta*sin_phi     +rnor()*Htherm
               Az = Hz-Nz*Ms*cos_theta+Hk*cos_theta+rnor()*Htherm

               dphi = gammap*(Ax*(-cos_theta*cos_phi - alpha*sin_phi) + Ay*(-cos_theta*sin_phi + alpha*cos_phi)&
                      + Az*sin_theta)/(sin_theta)+J_SHE*F*eta*(sin_phi-alpha*cos_phi*cos_theta)/(sin_theta&
                      * (1_dp+alpha**2)) - ((alpha*F*P*J_STT)/(1_dp+alpha**2))
               dtheta = gammap*(Ax*(alpha*cos_theta*cos_phi - sin_phi) + Ay*(alpha*cos_theta*sin_phi+cos_phi)&
                      - Az*alpha*sin_theta) - J_SHE*F*eta*(cos_phi*cos_theta + (alpha*sin_phi)/(1_dp+alpha**2))&
                      + ((F*P*J_STT)*sin_theta/(1_dp+alpha**2))
               ! only accounting for z-component
               R1 = Rp*(1_dp+(V/Vh)**2+TMR)/(1_dp+(V/Vh)**2 + TMR*(1_dp+cos_theta)/2_dp)

               ! Joule heating only operational for NYU device and is computed when J_STT is non zero
               if (J_STT .ne. 0.0_dp .and. heating_enabled) then
                  call heat_device(R1*(J_STT*A1)**2,t_i*t_step,T_init,T_free)
                  T = T_free
                  ! update temperature dependent parameters
                  call compute_K_and_Ms(K_295, Ms_295, T)
                  Bsat = Ms*u0
                  Hk = ((2.0_dp*Ki)/(tf*Ms*u0)) - ((2.0_dp*ksi*V)/(u0*Ms*tox*tf))
                  Htherm  = sqrt((2.0_dp*u0*alpha*kb*T)/(Bsat*gammab*t_step*volume))/u0
               elseif (heating_enabled) then
                  call cool_device(t_i*t_step,T_bath,T_init,T_free)
                  T = T_free
                  call compute_K_and_Ms(K_295, Ms_295, T)
                  Bsat = Ms*u0
                  Hk = ((2.0_dp*Ki)/(tf*Ms*u0)) - ((2.0_dp*ksi*V)/(u0*Ms*tox*tf))
                  Htherm = sqrt((2.0_dp*u0*alpha*kb*T)/(Bsat*gammab*t_step*volume))/u0
               end if

               pow = v_pow + she_pow + (R1*(J_STT*A1)**2)
               phi_i   = phi_i   + t_step*dphi
               theta_i = theta_i + t_step*dtheta
               cuml_pow = pow + cuml_pow
               if(fwrite_enabled) then
                  theta_evol(t_i) = theta_i
                   phi_evol(t_i)   = phi_i
                   temp_evol(t_i)  = T_free
               end if
           end do

        end subroutine drive

        subroutine file_dump(file_ID, phi_evol, theta_evol, temp_evol, overwrite_flag)
            implicit none
            integer,  parameter :: dp = kind(0.0d0)
            real(dp), dimension(:), intent(in) :: phi_evol, theta_evol, temp_evol
            integer,  intent(in) :: file_ID, overwrite_flag
            character(len=7) :: file_string

            ! ===== array dump to file of theta/phi time evolution  ====
            write (file_string,'(I7.7)') file_ID
            if( overwrite_flag .eq. 1) then
                open(unit = file_ID, file = "phi_time_evol_"//file_string//".txt", action = "write",&
                     status = "replace", form = 'formatted')
            else
                open(unit = file_ID, file = "phi_time_evol_"//file_string//".txt", action = "write",&
                     status = "old", position="append",form = 'formatted')
            end if
            write(file_ID,'(ES24.17)',advance="no") phi_evol
            close(file_ID)

            write (file_string,'(I7.7)') file_ID
            if( overwrite_flag .eq. 1) then
                open(unit = file_ID, file = "theta_time_evol_"//file_string//".txt", action = "write",&
                     status = "replace", form = 'formatted')
            else
                open(unit = file_ID, file = "theta_time_evol_"//file_string//".txt", action = "write",&
                     status = "old", position="append",form = 'formatted')
            end if
            write(file_ID,'(ES24.17)',advance="no") theta_evol
            close(file_ID)

            write (file_string,'(I7.7)') file_ID
            if( overwrite_flag .eq. 1) then
                open(unit = file_ID, file = "temp_time_evol_"//file_string//".txt", action = "write",&
                     status = "replace", form = 'formatted')
            else
                open(unit = file_ID, file = "temp_time_evol_"//file_string//".txt", action = "write",&
                     status = "old", position="append",form = 'formatted')
            end if
            write(file_ID,'(ES24.17)',advance="no") temp_evol
            close(file_ID)
        end subroutine file_dump
end module sampling
